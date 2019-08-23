import pygame
from pygame.locals import *
import hand_analyzer
import cv2
import numpy as np
import sys
import time
from handtracking.utils import detector_utils as detector_utils
from touch_arena import TouchArena

camera = cv2.VideoCapture(0)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
width = int(camera.get(3))
height = int(camera.get(4))

try:
    handset = hand_analyzer.PersistentHandset()
    initialized = False
    # Initialize an arena here
    ret, frame = camera.read()
    blurred_background = cv2.medianBlur(frame, 7)
    handset.background = blurred_background
    handset.bg_update = time.clock()
    canny_edges = cv2.Canny(frame, 40, 200, 3)
    blur_edges = cv2.GaussianBlur(canny_edges, (3, 3), 3)
    edge_contours, _ = cv2.findContours(blur_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, cnt in enumerate(edge_contours):
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        boxes.append(box)
    arena = TouchArena(width, height, boxes)

    while True:
        # Process Handset using opencv
        ret, frame = camera.read()
        boxes, scores = detector_utils.detect_objects(frame,
                                                      hand_analyzer.detection_graph, hand_analyzer.sess)
        blurred_frame = cv2.GaussianBlur(frame, (9, 9), 3)
        handset.update(boxes, scores, blurred_frame)

        # Debug window from opencv
        for hand in handset.hands:
            hand_analyzer.draw_hand_bb_on_image(frame, hand)
            cv2.circle(frame, hand.b_center, 10, (255, 0, 0), -1)
            # cv2.imshow("Mask", hand.mask)
            hullcopy = cv2.cvtColor(hand.mask.copy(), cv2.COLOR_GRAY2BGR)
            for point in hand.hull.tolist():
                p = point[0]
                cv2.circle(hullcopy, (p[0], p[1]), hand_analyzer.MINIMUM_TIP_RADIUS, (0, 0, 255), -1)
            # cv2.drawContours(hullcopy, [hand.hull], 0, (255, 0, 0), 1, 8)
            cv2.imshow("Hull", hullcopy)

        # Pygame uses RGB instead of BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update arena using hand info
        arena.add_hands(handset.hands)

        arena.hand_action_triggering(handset.hands)
        arena.event_input()
        arena.physics()
        arena.draw(frame)
        arena.cleanup()

        pygame.display.update()

        if not arena.running:
            break
except (KeyboardInterrupt, SystemExit):
    pygame.quit()
    cv2.destroyAllWindows()