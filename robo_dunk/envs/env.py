import math
import random
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces


@dataclass
class RoboDunkConfig:
    screen_width: int = 400
    screen_height: int = 400
    fps: int = 60

    # Robot
    robot_width: int = 50
    robot_height: int = 20
    robot_speed: int = 5
    arm_length: int = 80
    arm_min: int = 180
    arm_max: int = 360
    arm_speed: int = 4
    arm_elasticity: float = 2.0

    # Ball
    ball_radius: int = 8
    shoot_min_speed: int = 400
    shoot_max_speed: int = 500
    max_ball_speed: int = 600
    ball_elasticity: float = 1.0
    drag: float = 0.99

    # Cannon
    cannon_width: int = 6
    cannon_height: int = 30
    cannon_angle_min: int = 40
    cannon_angle_max: int = 50

    # Bucket
    bucket_height: int = 10
    bucket_width: int = 100
    bucket_x: int = 400
    bucket_y: int = 250

    # General
    object_elasticity: float = 0.5


class RoboDunkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        self.config = config or RoboDunkConfig()
        self.render_mode = render_mode
        self.screen_width = self.config.screen_width
        self.screen_height = self.config.screen_height
        self.clock = pygame.time.Clock()
        self.fps = self.config.fps

        self.action_space = spaces.MultiBinary(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 1),
            dtype=np.uint8,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 1),
            dtype=np.uint8,
        )

        self._setup()

    def seed(self, seed=None):
        """Seed RNGs for reproducible behavior."""
        self._seed = seed
        self.np_random, seed_ = gym.utils.seeding.np_random(seed)
        random.seed(seed)  # python random
        return [seed_]

    def _setup(self):
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Robo Dunk Bucket")
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.screen = None

        self.space = pymunk.Space()
        self.space.gravity = (0, 400)
        self.static_body = self.space.static_body

        # Walls
        walls = [
            pymunk.Segment(self.static_body, (0, 0), (0, self.screen_height), 2),
            pymunk.Segment(
                self.static_body,
                (self.screen_width, 0),
                (self.screen_width, self.screen_height),
                2,
            ),
            pymunk.Segment(self.static_body, (0, 0), (self.screen_width, 0), 2),
        ]
        for wall in walls:
            wall.elasticity = self.config.object_elasticity
            wall.friction = 0.5
        self.space.add(*walls)

        # Bucket
        self.bucket_x = self.config.bucket_x
        self.bucket_y = self.config.bucket_y
        self.bucket_width = self.config.bucket_width
        self.bucket_height = self.config.bucket_height
        self.bucket_floor = pymunk.Segment(
            self.static_body,
            (self.bucket_x - self.bucket_width, self.bucket_y),
            (self.bucket_x, self.bucket_y),
            3,
        )
        self.bucket_wall = pymunk.Segment(
            self.static_body,
            (self.bucket_x - self.bucket_width, self.bucket_y),
            (self.bucket_x - self.bucket_width, self.bucket_y - self.bucket_height),
            3,
        )
        self.bucket_floor.elasticity = self.config.object_elasticity
        self.bucket_wall.elasticity = self.config.object_elasticity
        self.space.add(self.bucket_floor, self.bucket_wall)

        # Robot
        self.robot_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.robot_body.position = (self.screen_width // 4, self.screen_height - 50)
        self.robot_shape = pymunk.Poly.create_box(
            self.robot_body, (self.config.robot_width, self.config.robot_height)
        )
        self.robot_shape.elasticity = self.config.object_elasticity
        self.space.add(self.robot_body, self.robot_shape)

        self.arm_angle = self.config.arm_min
        self.arm_shape = self._create_arm()
        self.space.add(self.arm_shape)

        # Cannon
        self.cannon_base = (self.screen_width - 40, self.screen_height - 50)
        self._spawn_ball()

        self.score = 0

    def _create_arm(self):
        start = (
            self.robot_body.position.x,
            self.robot_body.position.y - self.config.robot_height / 2,
        )
        end = (
            start[0] + self.config.arm_length * math.cos(math.radians(-self.arm_angle)),
            start[1] - self.config.arm_length * math.sin(math.radians(-self.arm_angle)),
        )
        arm = pymunk.Segment(self.static_body, start, end, 5)
        arm.elasticity = self.config.arm_elasticity
        arm.friction = 0.5
        return arm

    def _spawn_ball(self):
        # Use self.np_random if available, else fallback to random
        rng = getattr(self, "np_random", np.random)

        angle = rng.randint(self.config.cannon_angle_min, self.config.cannon_angle_max)
        speed = rng.randint(self.config.shoot_min_speed, self.config.shoot_max_speed)
        rad = math.radians(angle)

        tip_x = self.cannon_base[0] - self.config.cannon_height * math.cos(rad)
        tip_y = self.cannon_base[1] - self.config.cannon_height * math.sin(rad)

        self.ball_body = pymunk.Body(
            1, pymunk.moment_for_circle(1, 0, self.config.ball_radius)
        )
        self.ball_body.position = tip_x, tip_y
        self.ball_shape = pymunk.Circle(self.ball_body, self.config.ball_radius)
        self.ball_shape.elasticity = self.config.ball_elasticity
        self.ball_shape.friction = 0.5
        self.space.add(self.ball_body, self.ball_shape)

        self.ball_body.velocity = (-speed * math.cos(rad), -speed * math.sin(rad))
        self.ball_body.velocity_func = (
            lambda body, gravity, damping, dt: pymunk.Body.update_velocity(
                body, gravity, self.config.drag, dt
            )
        )
        self.next_cannon_angle = angle

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.space.remove(
            self.robot_shape, self.arm_shape, self.ball_body, self.ball_shape
        )
        self._setup()
        return self._get_obs(), {}

    def _get_obs(self):
        # Render the current state to an offscreen pygame surface
        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.fill((255, 255, 255))

        # Draw all objects (use same draw options as render)
        self.space.debug_draw(pymunk.pygame_util.DrawOptions(surface))

        # Convert pygame surface (W,H,3) → numpy array (H,W,3)
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))

        # Convert RGB to grayscale (still high-res, no resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Add channel dimension → (H, W, 1)
        obs = np.expand_dims(gray, axis=-1).astype(np.uint8)

        return obs

    def step(self, action):
        if action[0]:
            self.robot_body.position = (
                self.robot_body.position.x - self.config.robot_speed,
                self.robot_body.position.y,
            )
        if action[1]:
            self.robot_body.position = (
                self.robot_body.position.x + self.config.robot_speed,
                self.robot_body.position.y,
            )
        if action[2]:
            self.arm_angle = min(
                self.config.arm_max, self.arm_angle + self.config.arm_speed
            )
        if action[3]:
            self.arm_angle = max(
                self.config.arm_min, self.arm_angle - self.config.arm_speed
            )

        self.robot_body.position = (
            max(
                self.config.robot_width // 2,
                min(
                    self.screen_width // 2 - self.config.robot_width // 2,
                    self.robot_body.position.x,
                ),
            ),
            self.robot_body.position.y,
        )

        self.space.remove(self.arm_shape)
        self.arm_shape = self._create_arm()
        self.space.add(self.arm_shape)

        self.space.step(1 / self.fps)

        vx, vy = self.ball_body.velocity
        speed = (vx**2 + vy**2) ** 0.5
        if speed > self.config.max_ball_speed:
            scale = self.config.max_ball_speed / speed
            self.ball_body.velocity = vx * scale, vy * scale

        reward = 0
        terminated = False

        if (
            self.bucket_floor.point_query(self.ball_body.position).distance
            <= self.config.ball_radius
            and self.ball_body.velocity.y > 0
        ):
            reward = 10
            self.score += 1
            self.space.remove(self.ball_body, self.ball_shape)
            self._spawn_ball()
        elif self.ball_body.position.y > self.screen_height + 50:
            reward = -5
            self.space.remove(self.ball_body, self.ball_shape)
            self._spawn_ball()
        else:
            dist = abs(
                self.ball_body.position.x - (self.bucket_x - self.bucket_width / 2)
            )
            reward = max(0, 1 - dist / self.screen_width)

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return
        self.screen.fill((255, 255, 255))

        # Conveyor belt
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (0, self.screen_height - 40),
            (self.screen_width // 2, self.screen_height - 40),
            3,
        )

        # Cannon
        angle = math.radians(self.next_cannon_angle)
        offset = 5
        tip_x = self.cannon_base[0] - self.config.cannon_height * math.cos(angle)
        tip_y = self.cannon_base[1] - self.config.cannon_height * math.sin(angle)
        left_bar_start = (
            self.cannon_base[0] - offset * math.sin(angle),
            self.cannon_base[1] + offset * math.cos(angle),
        )
        left_bar_end = (
            tip_x - offset * math.sin(angle),
            tip_y + offset * math.cos(angle),
        )
        right_bar_start = (
            self.cannon_base[0] + offset * math.sin(angle),
            self.cannon_base[1] - offset * math.cos(angle),
        )
        right_bar_end = (
            tip_x + offset * math.sin(angle),
            tip_y - offset * math.cos(angle),
        )
        pygame.draw.line(self.screen, (34, 139, 34), left_bar_start, left_bar_end, 3)
        pygame.draw.line(self.screen, (34, 139, 34), right_bar_start, right_bar_end, 3)

        # Physics objects
        self.space.debug_draw(self.draw_options)

        # --- Draw the score bar ---
        font = pygame.font.Font(None, 28)  # default font, size 28
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))  # black text
        self.screen.blit(text, (10, 10))  # top-left corner

        # Update Display
        pygame.display.flip()
        self.clock.tick(self.config.fps)

    def close(self):
        pygame.quit()
