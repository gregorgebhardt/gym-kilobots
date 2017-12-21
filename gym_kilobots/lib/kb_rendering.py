import numpy as np
import pygame
from pygame import gfxdraw
pygame.init()


class KilobotsViewer(object):
    def __init__(self, width, height, caption=""):
        self._width = width
        self._height = height

        self._window = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(caption)
        pygame.event.set_allowed(pygame.QUIT)

        self._scale = np.eye(2)
        self._translation = np.zeros(2)

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scale_x = self._width / (right - left)
        scale_y = self._height / (top - bottom)
        self._scale = np.array([[scale_x, .0], [.0, scale_y]])
        self._translation = np.array([-left * scale_x, -bottom * scale_y])

    def _transform(self, position):
        return np.round(self._scale.dot(position) + self._translation).astype(int)

    def draw_aacircle(self, position=(0, 0), radius=.1, color=(0, 0, 0), filled=True, width=.01):
        position = self._transform(position)
        radius = int(self._scale[0, 0] * radius)
        width = 0 if filled else int(self._scale[0, 0] * width)

        pygame.draw.circle(self._window, color, position, radius, width)

    def draw_circle(self, position=(0, 0), radius=.1, color=(0, 0, 0), filled=True, width=.01):
        position = self._transform(position)
        radius = int(self._scale[0, 0] * radius)
        width = 0 if filled else int(self._scale[0, 0] * width)

        pygame.draw.circle(self._window, color, position, radius, width)

    def draw_polygon(self, vertices, color=(0, 0, 0), filled=True, width=.01):
        vertices = [self._transform(v) for v in vertices]
        width = 0 if filled else int(self._scale[0, 0] * width)
        # gfxdraw.aapolygon(self._window, vertices, color)
        # gfxdraw.filled_polygon(self._window, vertices, color)
        pygame.draw.polygon(self._window, color, vertices, 0 if filled else width)

    def draw_polyline(self, vertices, color=(0, 0, 0), closed=False, width=.01):
        vertices = [self._transform(v) for v in vertices]
        width = int(self._scale[0, 0] * width)
        pygame.draw.lines(self._window, color, closed, vertices, width)

    def draw_line(self, start, end, color=(0, 0, 0), width=.01):
        start = self._transform(start)
        end = self._transform(end)
        width = int(self._scale[0, 0] * width)
        pygame.draw.line(self._window, color, start, end, width)

    def get_array(self):
        image_data = pygame.surfarray.array3d(self._window)
        return image_data

    @staticmethod
    def render():
        pygame.display.flip()

    @staticmethod
    def close_requested():
        return pygame.event.peek(pygame.QUIT)

    @staticmethod
    def close():
        pygame.display.quit()
