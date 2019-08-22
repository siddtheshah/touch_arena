import numpy as np
import cv2
from handtracking.utils import detector_utils as detector_utils
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans
import math
import time
from collections import defaultdict

detection_graph, sess = detector_utils.load_inference_graph()

HAND_INNER_THRESH = 100
DETECTION_CONFIDENCE_THRESHOLD = .18
HAND_ASSIGNMENT_MAX_COST = 400
TIP_ASSIGNMENT_MAX_COST = 300
HAND_AREA_REMOVAL_THREHSOLD = .2
BOX_MINIMUM_AREA = 7200
PALM_RADIAL_COEFFICIENT = .6
MINIMUM_TIP_RADIUS = 15

tip_count = 0 # global variable. Not pretty, but useful for reconciliation
hand_count = 0 # also global

class Hand:
    def __init__(self, mask, box, color, b_center, contour):
        global hand_count
        self.mask = mask
        self.left, self.right, self.top, self.bottom = int(box[1]), int(box[3]), int(box[0]), int(box[2])
        self.color = color
        self.b_center = int(b_center[0]), int(b_center[1])
        self.contour = contour
        self.is_updated_from_nn = True  # Hands are only created if nn threshold is met.
        self.last_update = time.clock()
        self.tip_map = {}
        self.tip_decay = defaultdict(int)
        self.hand_id = hand_count
        hand_count += 1
        self.update()


    def update(self):

        # Update contour
        gray = cv2.convertScaleAbs(self.mask)
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contour = max(contours, key=cv2.contourArea)

        # Update and resolve hull
        concat = np.vstack(contours).squeeze()
        self.hull = cv2.convexHull(concat, False)
        self.resolve_hull()

        if not self.is_updated_from_nn:
            # Adjust the hand mask rectangle based on the MBR's movement
            x, y = bounding_center(self.mask)
            shift_x = self.b_center[0] - x
            shift_y = self.b_center[1] - y
            self.left, self.right, self.top, self.bottom = int(self.left - shift_x), int(self.right - shift_x), \
                                                           int(self.top - shift_y), int(self.bottom - shift_y)
            # bound adjustments inside frame
            if self.left < 2:
                self.left = 0
            elif self.left >= self.mask.shape[1] - 1:
                self.left = self.mask.shape[1] - 2
            if self.right >= self.mask.shape[1] - 2:
                self.right = self.mask.shape[1] - 1
            elif self.right < 2:
                self.right = 1
            if self.top < 2:
                self.top = 0
            elif self.top >= self.mask.shape[0] - 1:
                self.top = self.mask.shape[0] - 2
            if self.bottom >= self.mask.shape[0] - 2:
                self.bottom = self.mask.shape[0] - 1
            elif self.bottom < 2:
                self.bottom = 1
            self.b_center = x, y
        # using the mask, reconcile palm and fingertips
        M = cv2.moments(self.mask, 1)
        self.palm_center = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        self.palm_radius = math.sqrt(M["m00"]/math.pi)*PALM_RADIAL_COEFFICIENT

    def resolve_hull(self):
        global tip_count
        if not self.hull.size:
            return
        if not self.tip_map:
            for p in self.hull.tolist():
                self.tip_map[tip_count] = p[0]
                tip_count += 1
            return
        if self.hull.size > 15:
            mean_detections, score = kmeans(self.hull.squeeze().astype(float), 8)
            detections = [tuple(p) for p in mean_detections.tolist()]
        else:
            detections = [(p[0][0], p[0][1]) for p in self.hull.tolist()]
        cost_list = []
        tip_keys = list(self.tip_map.keys())
        for key in tip_keys:
            row = []
            tip = self.tip_map[key]
            for point in detections:
                row.append(l2_distance(tip, point))
            cost_list.append(row)
        cost = np.array(cost_list)
        row_ind, col_ind = linear_sum_assignment(cost)
        col_assignment = col_ind.tolist()
        row_assignment = row_ind.tolist()
        for detection_ind in range(len(detections)):
            if detection_ind not in col_assignment:
                self.tip_map[tip_count] = detections[detection_ind]
                tip_count += 1
        for key_ind in range(len(tip_keys)):
            if key_ind not in row_assignment:
                self.tip_decay[tip_keys[key_ind]] += 1
            else:
                self.tip_decay[tip_keys[key_ind]] = max(self.tip_decay[tip_keys[key_ind]] - 2, 0)
        for key_ind, detection_ind in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[key_ind, detection_ind] > TIP_ASSIGNMENT_MAX_COST:
                self.tip_map[tip_count] = detections[detection_ind]
                tip_count += 1
            else:
                self.tip_map[tip_keys[key_ind]] = detections[detection_ind]

        # Get rid of old detections
        to_delete = []
        for key in self.tip_decay.keys():
            if self.tip_decay[key] > 5:
                to_delete.append(key)
        for delete_this in to_delete:
            del self.tip_decay[delete_this]
            del self.tip_map[delete_this]

    def update_from_detection(self, detection):
        self.mask = detection.mask
        self.left = detection.left
        self.right = detection.right
        self.top = detection.top
        self.bottom = detection.bottom
        self.b_center = detection.b_center
        self.is_updated_from_nn = True
        self.color = detection.color
        self.last_update = time.clock()


    def dissimilarity(self, other):
        # Currently uses the nn box. Could also use model reconciliation cost.
        total = (self.left - other.left) + (self.right - other.right) + (self.top - other.top)
        total += (self.bottom - other.bottom)
        return total

class PersistentHandset:
    def __init__(self):
        self.hands = []
        self.background = None

    def adjust_or_remove_old_hands(self, blurred_frame):
        persisting_hands = []

        # mask update. Remove hand if mask doesn't have enough mass.
        for hand in self.hands:
            if hand.is_updated_from_nn:
                persisting_hands.append(hand)
                continue
            if time.clock() - hand.last_update > 8:
                continue
            # roi_rectangle = roi_rectangle_mask(blurred_frame, hand.left, hand.right, hand.top, hand.bottom)
            hand.color = get_primary_skin_color_from_roi(blurred_frame, hand.left, hand.right, hand.top, hand.bottom)
            # hand.mask = create_mask_using_inner_color(roi_rectangle, hand.color)
            full = create_mask_using_bg_subtraction(blurred_frame, self.background)
            hand.mask = roi_rectangle_mask(full, hand.left, hand.right, hand.top, hand.bottom)
            rectangle_area = abs(hand.left - hand.right)*abs(hand.top - hand.bottom)
            mask_area = np.sum(hand.mask > 0)

            if mask_area / rectangle_area < HAND_AREA_REMOVAL_THREHSOLD:
                print("Hand Area Too Low: ", mask_area / rectangle_area)
                continue
            persisting_hands.append(hand)

        # Update hands based on masks. Recalculate the palm-tip model, etc.
        self.hands = persisting_hands
        for hand in self.hands:
            hand.update()

    def resolve_detections(self):
        if not self.hands:
            self.hands = self.new_hands
            return
        if not self.new_hands:
            for hand in self.hands:
                hand.is_updated_from_nn = False
            return
        resolved = []
        cost_list = []
        for hand in self.hands:
            row = []
            for other in self.new_hands:
                row.append(hand.dissimilarity(other))
            cost_list.append(row)
        cost = np.array(cost_list)
        row_ind, col_ind = linear_sum_assignment(cost)
        row_assignment = row_ind.tolist()
        col_assignment = col_ind.tolist()
        for detection_ind in range(len(self.new_hands)):
            if detection_ind not in col_assignment:
                resolved.append(self.new_hands[detection_ind])
        for persistent_ind in range(len(self.hands)):
            if persistent_ind not in row_assignment:
                self.hands[persistent_ind].is_updated_from_nn = False
                resolved.append(self.hands[persistent_ind])
        for persistent_ind, detection_ind in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[persistent_ind, detection_ind] > HAND_ASSIGNMENT_MAX_COST:
                resolved.append(self.hands[persistent_ind])
                resolved.append(self.new_hands[detection_ind])
            else:
                self.hands[persistent_ind].update_from_detection(self.new_hands[detection_ind])
                resolved.append(self.hands[persistent_ind])

        self.hands = resolved
        return


    def add_detections(self, boxes, scores, blurred_frame):
        self.new_hands = []
        for box, score in zip(boxes, scores):
            if (score > DETECTION_CONFIDENCE_THRESHOLD):
                (left, right, top, bottom) = int(box[1]), int(box[3]), int(box[0]), int(box[2])
                rectangle_area = abs(left - right) * abs(top - bottom)
                if (rectangle_area > BOX_MINIMUM_AREA):
                    # roi_rectangle = roi_rectangle_mask(blurred_frame, left, right, top, bottom)
                    color = get_primary_skin_color_from_roi(blurred_frame, left, right, top, bottom)
                    # mask = create_mask_using_inner_color(roi_rectangle, color)
                    full = create_mask_using_bg_subtraction(blurred_frame, self.background)
                    mask = roi_rectangle_mask(full, left, right, top, bottom)
                    gray = cv2.convertScaleAbs(mask)
                    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        primary = max(contours, key=cv2.contourArea)
                        bc = bounding_center(mask)
                        self.new_hands.append(Hand(mask, box, color, bc, primary))
                    else:
                        print("No contour found")
                else:
                    print("Area requirement not met")

    def update(self,  boxes, scores, blurred_frame):
        self.add_detections(boxes, scores, blurred_frame)
        self.resolve_detections()
        self.adjust_or_remove_old_hands(blurred_frame)
        return

def l2_distance(point0, point1):
    return math.sqrt((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2)

def draw_hand_bb_on_image(frame, hand):
    if hand.is_updated_from_nn:
        color = (77, 255, 9)
    else:
        color = (230, 30, 30)
    cv2.rectangle(frame, (hand.left, hand.top), (hand.right, hand.bottom), color, 3, 1)


def bounding_center(mask):
    active_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(active_points)
    return int(x + w / 2), int(y + h / 2)

def roi_rectangle_mask(img, left, right, top, bottom):
    rectangle = np.zeros(img.shape, np.uint8)
    rectangle[top:bottom, left:right] = img[top:bottom, left:right]
    return rectangle

def get_primary_skin_color_from_roi(blurred_hand_image, left, right, top, bottom):
    inner_top = int(3*top/4 + bottom/4)
    inner_bottom = int(top/4 + 3*bottom/4)
    inner_left = int(right/4 + 3*left/4)
    inner_right = int(3*right/4 + left/4)
    inner_rect = blurred_hand_image[inner_top:inner_bottom, inner_left:inner_right]

    pixels = np.float32(inner_rect.reshape(-1, 3))
    n_colors = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    color = tuple(palette[np.argmax(counts)])

    return color

def create_mask_using_inner_color(roi_rectangle, color):
    diff = np.abs(roi_rectangle - color)
    sumdiff = np.sum(diff, 2)
    ret, mask = cv2.threshold(sumdiff, HAND_INNER_THRESH, 255, cv2.THRESH_BINARY_INV)

    return mask


def create_mask_using_bg_subtraction(raw_hand_image, blurred_background):
    blurred_hand = cv2.GaussianBlur(raw_hand_image, (11, 11), 3, 3)

    diff = cv2.absdiff(blurred_background, blurred_hand)
    diff_total = diff[:, :, 2] + diff[:, :, 1] + diff[:, :, 0]
    blur_diff = cv2.GaussianBlur(diff_total, (11, 11), 4, 4)
    ret, binary = cv2.threshold(blur_diff, 60, 255, cv2.THRESH_BINARY)
    return binary

def create_hybrid_mask(raw_hand_image, blurred_background, color):

    diff = np.abs(raw_hand_image - color)
    color_diff = np.sum(diff, 2)

    diff = cv2.absdiff(blurred_background, raw_hand_image)
    bg_diff = diff[:, :, 2] + diff[:, :, 1] + diff[:, :, 0]

def test_nn():
    camera = cv2.VideoCapture(0)
    handset = PersistentHandset()
    background_set = False

    while True:
        ret, frame = camera.read()

        if not background_set: # or (not handset.hands and time.clock() - handset.bg_update > 5):
            blurred_background = cv2.medianBlur(frame, 7)
            handset.background = blurred_background
            handset.bg_update = time.clock()
            background_set = True

        boxes, scores = detector_utils.detect_objects(frame,
                                                      detection_graph, sess)
        blurred_frame = cv2.GaussianBlur(frame, (9, 9), 3)
        handset.update(boxes, scores, blurred_frame)

        # print("Total Hands in Handset: ", len(handset.hands))

        # Debug views
        for hand in handset.hands:
            draw_hand_bb_on_image(frame, hand)
            cv2.circle(frame, hand.b_center, 10, (255, 0, 0), -1)
            cv2.imshow("Mask", hand.mask)
            hullcopy = cv2.cvtColor(hand.mask.copy(), cv2.COLOR_GRAY2BGR)
            for point in hand.hull.tolist():
                p = point[0]
                cv2.circle(hullcopy, (p[0], p[1]), MINIMUM_TIP_RADIUS, (0, 0, 255), -1)
            # cv2.drawContours(hullcopy, [hand.hull], 0, (255, 0, 0), 1, 8)
            cv2.imshow("Hull", hullcopy)
        cv2.imshow("Output", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

def test_bg_sub():
    camera = cv2.VideoCapture(0)
    background_set = False
    while True:
        ret, frame = camera.read()
        if not background_set:
            blurred_background = cv2.medianBlur(frame, 7)
            background_set = True

        frame = create_mask_using_bg_subtraction(frame, blurred_background)
        cv2.imshow("Output", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

if __name__ == "__main__":
    test_nn()