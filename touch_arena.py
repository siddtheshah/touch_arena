import pygame
from pygame.locals import *
from pygame.color import *

import pymunk as pm
from pymunk import Vec2d
import pymunk.util as u

COLLTYPE_DEFAULT = 0
COLLTYPE_FINGERTIP = 1
SQUARE_SPAWN = (140, 200)
SQUARE_LENGTH = 40
BALL_SPAWN = (140, 200)
BALL_DEFAULT_RADIUS = 20

TIP_RADIUS = 30
TARGET_FRAME_RATE = 24

def no_collide(arbiter, space, data):
    return False

def maintain_velocity(body, gravity, damping, dt):
    return

class TouchArena:
    def flipyv(self, v):
        return int(v.x), int(-v.y + self.h)

    def flip_point(self, v):
        return int(v[0]), int(-v[1] + self.h)

    def __init__(self, w, h, boxes):
        self.running = True
        ### Init pygame and create screen
        pygame.init()
        self.w, self.h = w, h
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        ### Init pymunk and create space
        self.space = pm.Space()
        self.space.gravity = (0.0, -900.0)
        h = self.space.add_collision_handler(COLLTYPE_FINGERTIP, COLLTYPE_FINGERTIP)
        h.begin = no_collide

        ### Walls
        self.walls = []
        self.create_wall_segments([(100, 50), (500, 50)])
        for box in boxes:
            b = box.tolist()
            flipped_box = [self.flip_point(p) for p in b]
            self.create_wall_segments(flipped_box)

        ## Balls
        # balls = [createBall(space, (100,300))]
        self.balls = []

        ## Fingertips
        self.fingertips = {}

        ## Palms
        self.palms = {}

        ### Polys
        self.polys = []


        # h = 10
        # for y in range(1, h):
        #     # for x in range(1, y):
        #     x = 0
        #     s = 10
        #     p = Vec2d(300, 40) + Vec2d(0, y * s * 2)
        #     self.polys.append(self.create_box(p, size=s, mass=1))

        self.run_physics = True

        ### Wall under construction
        self.wall_points = []
        ### Poly under construction
        self.poly_points = []

        self.shape_to_remove = None
        self.mouse_contact = None

    def create_ball(self, point, mass=4.0, radius=BALL_DEFAULT_RADIUS):

        moment = pm.moment_for_circle(mass, 0.0, radius)
        ball_body = pm.Body(mass, moment)
        ball_body.position = Vec2d(point)

        ball_shape = pm.Circle(ball_body, radius)
        ball_shape.friction = .5
        ball_shape.elasticity = .9999
        ball_shape.collision_type = COLLTYPE_DEFAULT
        self.space.add(ball_body, ball_shape)
        return ball_shape

    def create_box(self, pos, size=SQUARE_LENGTH, mass=10.0):
        box_points = [(-size, -size), (-size, size), (size, size), (size, -size)]
        return self.create_poly(box_points, mass=mass, pos=pos)

    def create_poly(self, points, mass=5.0, pos=(0, 0)):

        moment = pm.moment_for_poly(mass, points)
        # moment = 1000
        body = pm.Body(mass, moment)
        body.position = Vec2d(pos)
        shape = pm.Poly(body, points)
        shape.friction = 0.5
        shape.elasticity = .9999
        shape.collision_type = COLLTYPE_DEFAULT
        self.space.add(body, shape)
        return shape

    def create_wall_segments(self, points):
        """Create a number of wall segments connecting the points"""
        if len(points) < 2:
            return []
        points = list(map(Vec2d, points))
        for i in range(len(points) - 1):
            v1 = Vec2d(points[i].x, points[i].y)
            v2 = Vec2d(points[i + 1].x, points[i + 1].y)
            wall_body = pm.Body(body_type=pm.Body.STATIC)
            wall_shape = pm.Segment(wall_body, v1, v2, .0)
            wall_shape.friction = 1.0
            wall_shape.collision_type = COLLTYPE_DEFAULT
            self.space.add(wall_shape)
            self.walls.append(wall_shape)

    def create_fingertip(self, point, mass=1000.0, radius=TIP_RADIUS):
        moment = pm.moment_for_circle(mass, 0.0, radius)
        fingertip_body = pm.Body(mass, moment)
        fingertip_body.position = Vec2d(point)
        fingertip_body.velocity_func = maintain_velocity

        fingertip_shape = pm.Circle(fingertip_body, radius)
        fingertip_shape.friction = 20
        fingertip_shape.collision_type = COLLTYPE_FINGERTIP
        fingertip_shape.elasticity = .9999
        self.space.add(fingertip_body, fingertip_shape)
        return fingertip_shape

    def create_palm(self, point, radius):
        moment = pm.moment_for_circle(10000, 0.0, radius)
        palm_body = pm.Body(10000, moment)
        palm_body.position = Vec2d(point)
        palm_body.velocity_func = maintain_velocity

        palm_shape = pm.Circle(palm_body, radius)
        palm_shape.friction = 20
        palm_shape.collision_type = COLLTYPE_FINGERTIP
        palm_shape.elasticity = .5
        self.space.add(palm_body, palm_shape)
        return palm_shape

    def draw_ball(self, ball):
        body = ball.body
        v = body.position + ball.offset.cpvrotate(body.rotation_vector)
        p = self.flipyv(v)
        r = ball.radius
        pygame.draw.circle(self.screen, THECOLORS["blue"], p, int(r), 2)

    def draw_wall(self, wall):
        body = wall.body
        pv1 = self.flipyv(body.position + wall.a.cpvrotate(body.rotation_vector))
        pv2 = self.flipyv(body.position + wall.b.cpvrotate(body.rotation_vector))
        pygame.draw.lines(self.screen, THECOLORS["lightgray"], False, [pv1, pv2])


    def draw_poly(self, poly):
        body = poly.body
        ps = [p.rotated(body.angle) + body.position for p in poly.get_vertices()]
        ps.append(ps[0])
        ps = list(map(self.flipyv, ps))
        if u.is_clockwise(ps):
            color = THECOLORS["green"]
        else:
            color = THECOLORS["red"]
        pygame.draw.lines(self.screen, color, False, ps)

    def draw_fingertip(self, fingertip):
        body = fingertip.body
        v = body.position + fingertip.offset.cpvrotate(body.rotation_vector)
        p = self.flipyv(v)
        r = fingertip.radius
        pygame.draw.circle(self.screen, THECOLORS["red"], p, int(r), 2)

    def draw_palm(self, palm):
        body = palm.body
        v = body.position + palm.offset.cpvrotate(body.rotation_vector)
        p = self.flipyv(v)
        r = palm.radius
        if r > 2:
            pygame.draw.circle(self.screen, THECOLORS["orange"], p, int(r), 2)

    def draw(self, frame):
        # Fill in background.
        self.screen.fill([0, 0, 0])
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) REMEMBER COLOR CONVERSION IN MAIN LOOP
        frame = frame.swapaxes(0, 1)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))

        ### Draw balls
        for ball in self.balls:
            self.draw_ball(ball)

        ### Draw walls
        for wall in self.walls:
            self.draw_wall(wall)

        ### Draw polys
        for poly in self.polys:
            self.draw_poly(poly)

        ### Draw Hand Collision Surfaces
        for key, fingertip in self.fingertips.items():
            self.draw_fingertip(fingertip)
        for key, palm in self.palms.items():
            self.draw_palm(palm)

        ### All done, lets flip the display
        pygame.display.flip()

    def add_hands(self, persistent_hands):
        updated = []
        for hand in persistent_hands:
            for key, point in hand.tip_map.items():
                flipped = self.flipyv(Vec2d(point[0], point[1]))
                if key not in self.fingertips:
                    self.fingertips[key] = self.create_fingertip(flipped)
                else:
                    body = self.fingertips[key].body
                    body.position = flipped
                    body.velocity = 200*(flipped - body.position) #/TARGET_FRAME_RATE
                updated.append(key)
        all_keys = list(self.fingertips.keys())
        for key in all_keys:
            if key not in updated:
                self.space.remove(self.fingertips[key], self.fingertips[key].body)
                del self.fingertips[key]
        updated = []
        for hand in persistent_hands:
            flipped = self.flipyv(Vec2d(hand.palm_center[0], hand.palm_center[1]))
            if hand.hand_id not in self.palms:
                self.palms[hand.hand_id] = self.create_palm(flipped, hand.palm_radius)
            else:
                body = self.palms[hand.hand_id].body
                diff_x = flipped[0] - body.position[0]
                diff_y = flipped[1] - body.position[1]
                body.position = flipped
                body.velocity[0] = 50*diff_x #/TARGET_FRAME_RATE
                body.velocity[1] = 50*diff_y #/TARGET_FRAME_RATE
                self.palms[hand.hand_id].unsafe_set_radius(hand.palm_radius)
            updated.append(hand.hand_id)
        all_keys = list(self.palms.keys())
        for key in all_keys:
            if key not in updated:
                self.space.remove(self.palms[key], self.palms[key].body)
                del self.palms[key]


    def physics(self):
        if self.run_physics:
            x = 5  # Determines granularity of simulation
            dt = 1.0 / TARGET_FRAME_RATE / x
            for x in range(x):
                self.space.step(dt)
                for ball in self.balls:
                    pass
                for poly in self.polys:
                    pass

    def event_input(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_b:
                self.balls.append(self.create_ball(BALL_SPAWN))
            elif event.type == KEYDOWN and event.key == K_s:
                self.polys.append(self.create_box(SQUARE_SPAWN))
            elif event.type == KEYDOWN and event.key == K_SPACE:
                self.run_physics = not self.run_physics
            elif event.type == KEYDOWN and event.key == K_c:
                for poly in self.polys:
                    self.space.remove(poly, poly.body)
                self.polys = []
                for ball in self.balls:
                    self.space.remove(ball, ball.body)
                self.balls = []

    def cleanup(self):
        xs = []
        for ball in self.balls:
            if ball.body.position.x < -1000 or ball.body.position.x > 1000 \
                    or ball.body.position.y < -1000 or ball.body.position.y > 1000:
                xs.append(ball)
        for ball in xs:
            self.space.remove(ball, ball.body)
            self.balls.remove(ball)
        xs = []
        for poly in self.polys:
            if poly.body.position.x < -1000 or poly.body.position.x > 1000 \
                    or poly.body.position.y < -1000 or poly.body.position.y > 1000:
                xs.append(poly)
        for poly in xs:
            self.space.remove(poly, poly.body)
            self.polys.remove(poly)
