import math
import time

import cv2
import imutils
import numpy as np
import pymurapi as mur
from imutils.perspective import four_point_transform

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 1): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
}


class PD(object):
    """
    Реализация PD-контроллера.
    """
    _kp = 0.0
    _kd = 0.0
    _prev_error = 0.0
    _timestamp = 1

    def __init__(self):
        pass

    def set_p_gain(self, value):
        """
        Установка коэффициента пропорциональной составляющей.

        :param value: Значение коэффициента
        """
        self._kp = value

    def set_d_gain(self, value):
        """
        Установка коэффициента дифференцирования.

        :param value: Значение коэффициента
        """
        self._kd = value

    def process(self, error):
        """
        Расчёт выходного значения контроллера.

        :param error: Ошибка управления
        :return: Выходное значение контроллера
        """
        time.sleep(0.002)
        timestamp = int(round(time.time() * 1000))
        output = self._kp * error + self._kd / (timestamp - self._timestamp) * (error - self._prev_error)
        self._timestamp = timestamp
        self._prev_error = error
        return output


class Frame():
    """Класс для работы с кадром"""

    def __init__(self, image, robot):
        """Инициализация переменных экземпляра и обнаружение форм"""

        self.image = image
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.robot = robot

        # Определение цветов для форм
        self.define_colors()

        # Определение параметров для форм
        self.define_shape_parameters()

        # Обнаружение форм
        self.detect_shapes()

    def define_colors(self):
        """Определение диапазонов цветов для каждой формы"""

        self.circle_color = ([5, 150, 150], [20, 255, 255])
        self.sensors_color = ([20, 190, 150], [30, 255, 255])
        self.broken_sensors_color = ([0, 0, 0], [180, 255, 100])
        self.course_line_color = (([0, 180, 50], [10, 255, 255]), ([170, 180, 50], [180, 255, 255]))

    def define_shape_parameters(self):
        """Определение параметров форм"""

        self.circle_min_area = 500
        self.circle_match_ratio = 0.7
        self.sensor_min_area = 500
        self.sensor_max_area = 200000

    def detect_shapes(self):
        """Обнаружение каждой формы и присвоение каждой из них словарю"""

        # Обнаружение кругов
        circle_detected = self.detect_circle()
        self.circle = {'center': circle_detected[1], 'radius': circle_detected[2]} if circle_detected[0] else None

        # Обнаружение датчиков
        sensor_detected = self.detect_sensor()
        self.sensor = {'type': sensor_detected[1], 'center': sensor_detected[2]} if sensor_detected[0] else None

        # Обнаружение неисправных датчиков
        broken_sensor_detected = self.detect_broken_sensor()
        self.broken_sensor = {'center': broken_sensor_detected[1], 'contour': broken_sensor_detected[2]} if \
            broken_sensor_detected[0] else None

        # Обнаружение линии движения
        course_line_detected = self.detect_course_line()
        self.course_line = {'center': course_line_detected[1], 'angle': course_line_detected[2]} if \
            course_line_detected[0] else None

    def get_mask(self, color):
        """Создает маску для определенного диапазона цветов"""
        lower = np.array(color[0])
        upper = np.array(color[1])
        return cv2.inRange(self.hsv, lower, upper)

    def find_contours(self, img, color=None):
        """
        Находит контуры на изображении (или маске).

        :param img: Изображение или маска
        :param color: Диапазон цвета (опционально)
        :returns: Контуры
        """
        mask = img if color is None else self.get_mask(color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_shape(self, color, min_area=None, sensor=False):
        """
        Универсальная функция для обнаружения формы на изображении.

        :return: Кортеж из булевого значения (True, если форма найдена), центра формы и
        контура или радиуса.
        """
        contours = self.find_contours(self.hsv, color)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            area = cv2.contourArea(cnt)
            if sensor and min_area <= area <= self.sensor_max_area:
                return True, center, cnt
            elif not sensor and area >= self.circle_match_ratio * radius ** 2 * math.pi > min_area:
                return True, center, round(radius, 1)

        return False, None, None

    def detect_circle(self):
        """
        Функция обнаружения кружка на изображении.
        """
        return self.detect_shape(self.circle_color, self.circle_min_area)

    def detect_broken_sensor(self):
        """
        Распознает сломанные сенсоры на изображении
        """
        return self.detect_shape(self.broken_sensors_color, self.sensor_min_area, sensor=True)

    def detect_sensor(self):
        """
        Распознает почти все объекты на изображении, возвращает тип объекта и его центр

        :return: Кортеж (True, тип, центр объекта), если объект найден, иначе (False, None, None).
        """
        contours = self.find_contours(self.hsv, self.sensors_color)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            center, _ = cv2.minEnclosingCircle(cnt)
            area = cv2.contourArea(cnt)
            if not self.sensor_min_area < area < self.sensor_max_area:
                return False, None, None
            center = (int(center[0]), int(center[1]))

            rectangle = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rectangle))
            rectangle_area = cv2.contourArea(box)

            try:
                triangle = np.int0(cv2.minEnclosingTriangle(cnt)[1])
                triangle_area = cv2.contourArea(triangle)
            except:
                triangle_area = 0

            shape_areas = {
                'rectangle': rectangle_area,
                'triangle': triangle_area
            }

            diffs = {name: abs(area - shape_areas[name]) for name in shape_areas.keys()}
            shape_name = min(diffs, key=diffs.get)

            return True, shape_name, center

        return False, None, None

    def detect_course_line(self):
        """
        Распознает линию курса на изображении

        :return: Кортеж (True, центр линии, угол), если линия найдена, иначе (False, None, None).
        """

        # Объединение масок для красного цвета
        mask = self.get_mask(self.course_line_color[0]) + self.get_mask(self.course_line_color[1])

        # Поиск контуров на маске
        contours = self.find_contours(mask)

        # Если контуры не найдены, вернуть False
        if not contours:
            return False, None, None

        # Выбор контура с максимальной областью
        cnt = max(contours, key=cv2.contourArea)
        # Поиск прямоугольника минимальной площади, который заключает контур
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Сортировка точек по расстоянию между ними
        sorted_points = sorted(box, key=lambda x: np.linalg.norm(box[0] - x))

        # Найдите середины меньших сторон
        midpoint1 = ((sorted_points[0][0] + sorted_points[1][0]) // 2, (sorted_points[0][1] + sorted_points[1][1]) // 2)
        midpoint2 = ((sorted_points[2][0] + sorted_points[3][0]) // 2, (sorted_points[2][1] + sorted_points[3][1]) // 2)

        # Находим угол между линией и вертикальной осью
        dy = midpoint2[1] - midpoint1[1]
        dx = midpoint2[0] - midpoint1[0]
        angle = round((math.atan2(dy, dx) * 180. / math.pi) + 90, 2)
        return True, (int(rect[0][0]), int(rect[0][1])), angle

    def detect_digit(self):
        """Функция для распознавания цифр на изображении.
        Возвращает кортеж с информацией о найденной цифре или None.

        :return: (True, центр цифры, цифра, угол рамки), если цифра найдена, иначе (False, центр изображения, None, None).
        """

        # Применение функций обработки изображения
        image = cv2.copyMakeBorder(self.image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized = imutils.resize(image, height=500)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        mask = cv2.inRange(blurred, np.array([0, 0, 220]), np.array([180, 20, 255]))
        edged = cv2.Canny(mask, 50, 200, 255)

        # Использование компактного представления контуров для уменьшения использования памяти
        contours = imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE))
        contours = sorted(contours, key=cv2.contourArea, reverse=False)

        displayCnt = None
        center = (0, 0)
        rect = None

        # Модификация цикла для уменьшения количества итераций
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            # Прекращаем итерацию, когда находим подходящий контур
            if len(approx) == 4:
                displayCnt = approx
                center, _ = cv2.minEnclosingCircle(contour)
                rect = cv2.minAreaRect(contour)
                break

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Сортировка точек вместо использования цикла - для увеличения скорости
        sorted_points = sorted(box, key=lambda x: np.linalg.norm(box[0] - x))

        # Вычисляем середины меньших сторон и угол между линией и вертикальной осью
        midpoint1 = ((sorted_points[0][0] + sorted_points[1][0]) // 2, (sorted_points[0][1] + sorted_points[1][1]) // 2)
        midpoint2 = ((sorted_points[2][0] + sorted_points[3][0]) // 2, (sorted_points[2][1] + sorted_points[3][1]) // 2)
        dy = midpoint2[1] - midpoint1[1]
        dx = midpoint2[0] - midpoint1[0]
        angle = round(math.degrees(math.atan2(dy, dx)) + 90, 2)

        try:
            # Применение преобразования перспективы и пороговую обработку
            # к искаженному изображению вместо использования цикла
            warped = four_point_transform(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), displayCnt.reshape(4, 2))
            output = four_point_transform(image, displayCnt.reshape(4, 2))
            thresh = cv2.morphologyEx(cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
                                      cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5)))

            # Проанализировать каждую цифру на пороговом изображении
            contours = imutils.grab_contours(
                cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

            for digit in contours:
                # Определить область интереса (ROI) цифры и рассчитать параметры сегмента
                (x, y, w, h) = cv2.boundingRect(digit)
                roi = thresh[y:y + h, x:x + w]
                (roiH, roiW) = roi.shape
                (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
                dHC = int(roiH * 0.05)

                # Определить набор из 7 сегментов
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                segments = [((0, 0), (w, dH)),  # верх
                            ((0, 0), (dW, h // 2)),  # верхний левый
                            ((w - dW, 0), (w, h // 2)),  # верхний правый
                            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # центр
                            ((0, h // 2), (dW, h)),  # нижний левый
                            ((w - dW, h // 2), (w, h)),  # нижний правый
                            ((0, h - dH), (w, h))  # низ
                            ]

                # Проанализировать каждый сегмент
                on = [cv2.countNonZero(roi[yA:yB, xA:xB]) / float((xB - xA) * (yB - yA)) > 0.5 for
                      (i, ((xA, yA), (xB, yB))) in enumerate(segments)]

                # Распознаем цифру и возвращаем результат
                digit = DIGITS_LOOKUP[tuple(on)]
                return True, center, digit, angle
        except:
            print('error')
            cv2.imwrite('errorimage.jpg', self.image)
            return False, (320 / 2, 240 / 2), None, None

    def put_info(self):
        """
        Добавляет информацию о различных объектах на изображение.
        """
        # Универсализация процесса добавления текста на изображение
        objects = {'Circle': self.circle, 'Sensor': self.sensor,
                   'Broken': self.broken_sensor, "Line {}".format(self.course_line['angle']): self.course_line}

        for obj_name, obj in objects.items():
            if obj is not None:
                cv2.putText(self.image, obj_name, obj['center'], cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 100), 1)

    def get_result(self):
        """
        Возвращает информацию об объектах на изображении в виде словаря.
        """
        return {
            'circle': self.circle,
            'sensor': self.sensor,
            'broken_sensor': self.broken_sensor,
            'course_line': self.course_line
        }

    def show_frame(self, cam, mode):
        """
        Отображает текущее изображение.

        :param cam: Индекс камеры.
        :param mode: Режим отображения (True для использования mur_view, False для использования cv2).
        """
        if mode:
            self.robot.mur_view.show(self.image, cam)
        else:
            cv2.imshow('Camera' + str(cam), self.image)
            if cv2.waitKey(10) & 0xFF == 27:  # ESC закрывает окно
                pass


class Motor:
    """ Класс представляет двигатель АУВ """

    def __init__(self, auv, motor_id, delta=0, coefficient=0):
        """ Конструктор

        Args:
            auv: объект АУВ, которому этот двигатель принадлежит
            motor_id: идентификатор двигателя
            delta: смещение, используемое при калибровке
            coefficient: коэффициент, используемый при калибровке
        """
        self.motor_id = motor_id
        self.auv = auv
        self.motor_power = 0  # current motor power
        self.max_power = 100  # maximum motor power
        self.motor_direction = 1  # motor direction (1 - forward, -1 - backward)
        self.motor_calibration_delta = delta
        self.motor_calibration_coefficient = coefficient

    def set_motor_power(self, new_power):
        """ Установить мощность двигателя

        Args:
            new_power: новая мощность двигателя
        """
        self.motor_direction = 1 if new_power >= 0 else -1
        self.motor_power = min(abs(new_power), self.max_power)

    def set_calibration_parameters(self, new_delta, new_coefficient):
        """ Установить параметры калибровки

        Args:
            new_delta: новое смещение
            new_coefficient: новый коэффициент
        """
        self.motor_calibration_delta = new_delta
        self.motor_calibration_coefficient = new_coefficient

    def set_max_power(self, new_max_power):
        """ Установить максимальную мощность

        Args:
            new_max_power: новая максимальная мощность
        """
        self.max_power = new_max_power

    def run_motor(self):
        """ Запустить двигатель """
        self.auv.set_motor_power(self.motor_id, self.motor_power * self.motor_direction)

    def stop_motor(self):
        """ Остановить двигатель """
        self.set_motor_power(0)
        self.run_motor()


class Robot:
    """Класс, представляющий робота."""

    def __init__(self):
        # Инициализация системы робота и определение начальных параметров
        self.auv = mur.mur_init()
        self.yaw = self.auv.get_yaw()
        self.depth = self.auv.get_depth()
        self.roll = self.auv.get_roll()
        self.speed_forward = 0
        self.speed_side = 0
        self.prev_update_time = 0
        self.f_image = None
        self.b_image = None
        self.real = None
        self.missions = []

        self.determine_environment()

        self.initialize_regulators()

        self.initialize_motors()

    def determine_environment(self):
        """Определение среды (реальный робот или симулятор)"""
        try:
            self.mur_view = self.auv.get_videoserver()
            self.cap0 = cv2.VideoCapture(1)
            self.cap1 = cv2.VideoCapture(0)
            self.real = True
        except:
            self.real = False

    def initialize_regulators(self):
        """Инициализация и настройка регуляторов"""
        self.yaw_regulator = self.init_regulator(0.8, 0.5)
        self.depth_regulator = self.init_regulator(40, 50)
        self.stab_X_regulator = self.init_regulator(0.8, 10)
        self.stab_Y_regulator = self.init_regulator(0.8, 10)

    def init_regulator(self, p_gain, d_gain):
        """Вспомогательная функция для инициализации и настройки регулятора"""
        regulator = PD()
        regulator.set_p_gain(p_gain)
        regulator.set_d_gain(d_gain)
        return regulator

    def initialize_motors(self):
        """Инициализация моторов в зависимости от среды"""
        if self.real:
            motor_ids = [1, 2, 3, 0]
        else:
            motor_ids = [0, 1, 2, 3]

        self.motor_X_l = self.init_motor(motor_ids[0])
        self.motor_X_r = self.init_motor(motor_ids[1])
        self.motor_Z_l = self.init_motor(motor_ids[2])
        self.motor_Z_r = self.init_motor(motor_ids[3])

        if not self.real:
            self.motor_Y = Motor(self.auv, 4)

    def init_motor(self, motor_id):
        """Вспомогательная функция для инициализации мотора"""
        motor = Motor(self.auv, motor_id)
        motor.set_calibration_parameters(0, 0)
        return motor

    @staticmethod
    def clamp(value, min_value, max_value):
        """Ограничивает значение в заданном диапазоне.

        :param value: Значение для ограничения.
        :param min_value: Минимальное значение диапазона.
        :param max_value: Максимальное значение диапазона.
        :returns: Value, ограниченное в диапазоне [min_value, max_value].
        """
        return max(min_value, min(value, max_value))  # оптимизированное ограничение значения

    @staticmethod
    def clamp_to180(angle):
        """Приводит угол до значения от -180 до 180.

        :param angle: Исходный угол.
        :returns: Преобразованный угол.
        """
        return (angle + 180) % 360 - 180  # оптимизированное приведение угла

    # Методы установки значений
    def set_yaw(self, value):
        self.yaw = value

    def set_depth(self, value):
        self.depth = value

    def set_speed_forward(self, value):
        self.speed_forward = value

    def set_speed_side(self, value):
        self.speed_side = value

    # Методы получения значений
    def get_yaw(self):
        return self.yaw

    def get_depth(self):
        return self.depth

    def get_speed_forward(self):
        return self.speed_forward

    def get_speed_side(self):
        return self.speed_side

    # Методы работы с миссиями
    def add_mission(self, func):
        """Добавляет новую миссию в конец очереди миссий.

        :param func: Функция (миссия), которую нужно добавить в очередь.
        """
        self.missions.append(func)

    def pop_mission(self):
        """Извлекает и возвращает первую миссию из очереди миссий."""
        return self.missions.pop(0) if self.missions else None  # обработка случая пустой очереди

    def get_missions_length(self):
        """Возвращает количество миссий в очереди."""
        return len(self.missions)

    def set_motor_power(self, power, *motors):
        """Устанавливает мощность для указанных моторов.

        :param power: Мощность для установки.
        :param motors: Моторы для установки мощности.
        """
        for motor in motors:
            motor.set_motor_power(power)

    def update(self):
        """Обновляет состояние робота, поддерживая заданную глубину и курс."""
        self.get_image(2)
        self.keep_depth(self.depth)
        self.keep_yaw(self.yaw, self.speed_forward)
        self.set_motor_power(self.speed_side, self.motor_Y)
        self.run_all_motors()

    def keep_depth(self, depth_to_set):
        """Поддерживает заданную глубину.

        :param depth_to_set: Глубина для поддержания.
        """
        error = round(self.auv.get_depth() - depth_to_set, 2)
        output = self.depth_regulator.process(error)
        self.set_motor_power(round(output, 2), self.motor_Z_l, self.motor_Z_r)

    def keep_yaw(self, yaw_to_set, speed):
        """Поддерживает заданный курс и скорость.

        :param yaw_to_set: Курс для поддержания.
        :param speed: Скорость для поддержания.
        """
        error = self.auv.get_yaw() - yaw_to_set
        error = round(self.clamp_to180(error), 2)
        output = self.yaw_regulator.process(error)
        power = round(output, 2)
        self.set_motor_power(-power + speed, self.motor_X_l)
        self.set_motor_power(power + speed, self.motor_X_r)

    def run_all_motors(self):
        """Запускает все моторы робота."""
        for motor in (self.motor_X_l, self.motor_X_r, self.motor_Z_l, self.motor_Z_r, self.motor_Y):
            motor.run_motor()

    def stop_motors(self):
        """Останавливает все моторы и устанавливает скорость в ноль."""
        self.set_motor_power(0, self.motor_X_l, self.motor_X_r, self.motor_Z_l, self.motor_Z_r, self.motor_Y)
        self.run_all_motors()
        self.set_speed_forward(0)
        self.set_speed_side(0)

    def create_image(self, frame):
        """Создает изображение из кадра.

        :param frame: Кадр для создания изображения.
        """
        return Frame(frame, self)

    def get_image(self, cams):
        """Получает изображение с камеры.

        :param cams: Определяет, какие камеры следует использовать.
        """
        if cams == 0:
            frame1 = self.get_frame(cap0=True)
            self.f_image = self.create_image(frame1)
        elif cams == 1:
            frame2 = self.get_frame(cap1=True)
            self.b_image = self.create_image(frame2)
        elif cams == 2:
            frame1 = self.get_frame(cap0=True)
            frame2 = self.get_frame(cap1=True)
            self.f_image = self.create_image(frame1)
            self.b_image = self.create_image(frame2)

    def get_frame(self, cap0=False, cap1=False):
        """Получает кадр из камеры.

        :param cap0: Если истина, используется cap0.
        :param cap1: Если истина, используется cap1.
        """
        if self.real:
            if cap0:
                _, frame = self.cap0.read()
            elif cap1:
                _, frame = self.cap1.read()
        else:
            if cap0:
                frame = self.auv.get_image_front()
            elif cap1:
                frame = self.auv.get_image_bottom()
        return frame

    def show_image(self, cams, put=False):
        """Отображает изображение с камеры.

        :param cams: Определяет, какие камеры следует использовать.
        :param put: Определяет, следует ли добавить информацию к изображению.
        """
        if put:
            self.add_info(cams)

        if cams in (0, 2):
            self.f_image.show_frame(0, self.real)
        if cams in (1, 2):
            self.b_image.show_frame(1, self.real)

    def add_info(self, cams):
        """Добавляет информацию к изображению.

        :param cams: Определяет, какие камеры следует использовать.
        """
        if cams in (0, 2):
            self.f_image.put_info()
        if cams in (1, 2):
            self.b_image.put_info()

    def stab_on_object(self):
        # Получаем результаты от изображения (детектированные объекты)
        objects = self.b_image.get_result()

        # Инициализация переменных для проверки наличия объектов и их центров
        found = False
        broken = False
        center = 0, 0

        # Проверяем наличие объектов типов 'circle', 'sensor', 'broken_sensor' и устанавливаем статус found в True,
        # если они найдены
        if objects['circle'] is not None:
            found = True
            center = objects['circle']['center']
        elif objects['sensor'] is not None:
            found = True
            center = objects['sensor']['center']
        elif objects['broken_sensor'] is not None:
            found = True
            broken = True
            center = objects['broken_sensor']['center']

        # Если объект найден
        if found:
            # Вычисление координат центра объекта относительно центра изображения (с учетом небольшого смещения для
            # 'broken_sensor')
            x_center = center[0] - (320 / 2)
            y_center = center[1] - (240 / 2) - 25
            if broken:
                y_center += 25

            # Регулирование скорости по оси X и Y, используя соответствующие регуляторы stab_X и stab_Y
            self.speed_forward = self.clamp(-self.stab_X_regulator.process(y_center), -10, 10)
            self.speed_side = self.clamp(-self.stab_Y_regulator.process(x_center), -10,
                                         10) / 2 if broken else self.clamp(-self.stab_Y_regulator.process(x_center),
                                                                           -10, 10)

            # Определение расстояния от центра изображения до объекта
            length = math.sqrt(x_center ** 2 + y_center ** 2)

            # Возврат True, если объект близко к центру (для 'broken_sensor' менее 1.0, для остальных менее 25.0)
            if (broken and length < 10.0) or (length < 80.0 and not broken):
                return True

        # Если никакой объект не найден - возвращаем False
        return False

    def do_sensor_task(self, sensor_type):
        # Если тип датчика 'broken', аппарат производит действие drop() и возвращает True
        if sensor_type == 'broken':
            self.auv.drop()
            return True
        # Если тип датчика 'triangle', двигатели вращаются в противоположных направлениях на 4 секунды, затем они
        # возвращаются к исходной скорости
        elif sensor_type == 'triangle':
            self.motor_X_l.set_motor_power(60)
            self.motor_X_r.set_motor_power(-60)
            self.motor_X_l.run_motor()
            self.motor_X_r.run_motor()
            time.sleep(3.7)
            speed = self.get_speed_forward()
            self.set_speed_forward(speed)
            return True
        # Если тип датчика 'rectangle', аппарат сначала меняет свою глубину на 3.9 метра, затем удерживает эту глубину,
        # двигается вперед со скоростью 20, после чего он возвращается к исходной скорости
        elif sensor_type == 'rectangle':
            self.set_depth(3.9)
            speed = self.get_speed_forward()
            while abs(self.depth - self.auv.get_depth()) > 0.15:
                self.set_speed_forward(0)
                self.stab_on_object()
                self.update()
            self.set_depth(3.2)
            self.keep_depth(self.depth)
            self.set_speed_forward(15)
            while abs(self.depth - self.auv.get_depth()) > 0.35:
                self.update()
            self.set_speed_forward(speed)
            return True
        # Если тип датчика не соответствует ни одному из указанных, возвращается False
        return False

    def walk_the_line(self):
        # Обновление текущего состояния
        self.update()

        # Получение информации об объектах вокруг
        objects = self.b_image.get_result()

        # Флаг, указывающий на наличие целевых объектов
        found = False

        # Координаты центра целевого объекта
        center = 0, 0

        # Проверка наличия объектов различных типов и определение их центра
        if objects['circle'] is not None:
            found = True
            center = objects['circle']['center']
        elif objects['sensor'] is not None:
            found = True
            center = objects['sensor']['center']
        elif objects['broken_sensor'] is not None:
            found = True
            center = objects['broken_sensor']['center']

        # Вычисление расстояния между центром объекта и подразумеваемым центром изображения
        length = center[1] - (240 / 2 - 75)
        length = length * 1.8

        # Если объект найден и находится близко, завершить функцию
        if found and length < 1.0:
            return True

        # Определение линии курса
        line = objects['course_line']

        # Пытаемся определить центр линии курса. Если ошибка, завершить функцию
        try:
            center = line['center']
            # Определение угла между линией курса и горизонтом
            angle = line['angle']

            # Расчет нового угла на основе текущего угла автономного подводного аппарата (AUV) и угла линии курса
            new_angle = self.auv.get_yaw() + angle

            # Установка нового угла для движения AUV
            self.keep_yaw(new_angle, self.get_speed_forward())
            self.set_yaw(new_angle)

            # Определение отклонения центра линии курса от центра изображения по оси X
            x_center = (center[0] - (320 / 2))

            # Расчет боковой скорости для стабилизации AUV на курсе
            self.speed_side = -self.stab_Y_regulator.process(x_center)
        except:
            return True
        return False

    def dive_to_depths(self):

        # Установка глубины 2.4 метра для погружения подводного аппарата
        self.set_depth(2.4)

        # Цикл обновления информации до достижения требуемой глубины с заданной точностью
        while abs(self.depth - self.auv.get_depth()) > 0.01:
            self.update()

        # После достижения первой глубины, устанавливаем глубину 3.1 метра
        self.set_depth(3.1)

        # Обновляем информацию до достижения требуемой глубины с указанной точностью ИЛИ пока стабилизация на объекте
        # не будет достигнута
        while abs(self.depth - self.auv.get_depth()) > 0.01 or not self.stab_on_object:
            self.update()

        # Затем устанавливаем глубину 3.2 метра
        self.set_depth(3.2)

        return True

    def go_to_do_task(self):
        # Получаем угол относительно курсовой линии
        angle = self.b_image.get_result()['course_line']['angle']

        # Устанавливаем новый курс, суммируя текущий угол AUV и угол курсовой линии
        self.set_yaw(self.auv.get_yaw() + angle)

        # Начинаем движение вперед со скоростью 20
        self.set_speed_forward(20)

        # Если еще не достигли линии, продолжаем обновлять информацию
        while not self.walk_the_line():
            self.update()

        # Если объект еще не стабилизирован, продолжаем обновлять информацию
        while not self.stab_on_object():
            self.walk_the_line()
            self.update()

        # Получаем информацию об объектах вокруг
        objects = self.b_image.get_result()

        # Если есть работающий датчик, выполняем задачу для него
        if objects['sensor'] is not None:
            while not self.do_sensor_task(objects['sensor']['type']):
                self.update()

        # Если есть сломанный датчик, выполняем задачу для него
        elif objects['broken_sensor'] is not None:
            while not self.do_sensor_task('broken'):
                self.update()

        return True

    def go_to_number(self):
        # Устанавливаем глубину погружения
        self.set_depth(3.25)

        # Устанавливаем скорость движения вперед
        self.set_speed_forward(20)

        # Если еще не достигли линии, продолжаем обновлять информацию
        while not self.walk_the_line():
            self.update()

        # Если объект еще не стабилизирован, продолжаем обновлять информацию
        while not self.stab_on_object():
            self.update()

        # Останавливаем движение вперед
        self.set_speed_forward(0)

        # Обнаруживаем число на b-изображении и f-изображении
        b_digit = self.b_image.detect_digit()
        f_digit = self.f_image.detect_digit()

        # Расчет угла между AUV и обнаруженным числом на b-изображении
        self.yaw = self.auv.get_yaw() + b_digit[3]

        # Определение числа на b-изображении и f-изображении
        b_num = b_digit[2]
        f_num = f_digit[2]
        # Если числа на b-изображении и f-изображении не совпадают
        if b_num != f_num:

            # Если число на b-изображении не равно 1 или 2
            if b_num != 1 and b_num != 2:

                # Устанавливаем глубину 2.85 метра
                self.set_depth(2.85)

                # Цикл стабилизации на объекте до достижения требуемой глубины с заданной точностью
                while abs(self.depth - self.auv.get_depth()) > 0.05:
                    self.stab_on_object()
                    self.update()

                # Повторное обнаружение числа на f-изображении
                f_digit = self.f_image.detect_digit()
                f_num = f_digit[2]

                # Если числа все еще не совпадают
                if b_num != f_num:
                    self.set_depth(3.65)
                    while abs(self.depth - self.auv.get_depth()) > 0.05:
                        self.stab_on_object()
                        self.update()

                    # Повторное обнаружение числа на f-изображении
                    f_digit = self.f_image.detect_digit()
                    f_num = f_digit[2]

                    # Если числа все еще не совпадают
                    if b_num != f_num:
                        print('ячейка не найдена')
                        return True

            # Если число на b-изображении не равно 3
            if b_num != 3:

                # Устанавливаем глубину 2.85 метра
                self.set_depth(3.65)

                # Цикл стабилизации на объекте до достижения требуемой глубины с заданной точностью
                while abs(self.depth - self.auv.get_depth()) > 0.05:
                    self.stab_on_object()
                    self.update()

                # Повторное обнаружение числа на f-изображении
                f_digit = self.f_image.detect_digit()
                f_num = f_digit[2]

                # Если числа все еще не совпадают
                if b_num != f_num:
                    self.set_depth(2.85)
                    while abs(self.depth - self.auv.get_depth()) > 0.05:
                        self.stab_on_object()
                        self.update()

                    # Повторное обнаружение числа на f-изображении
                    f_digit = self.f_image.detect_digit()
                    f_num = f_digit[2]

                    # Если числа все еще не совпадают
                    if b_num != f_num:
                        print('ячейка не найдена')
                        return True

        # Если объект еще не стабилизирован, продолжаем обновлять информацию
        while not self.stab_on_object():
            self.update()

        # Запускаем движение вперед со скоростью 20
        self.set_speed_forward(20)

        # Цикл обновления информации в течение 3 секунд
        start_time = time.time()
        while time.time() - start_time < 3:
            self.update()

        return True

    def run(self):
        # Устанавливаем начальный курс, глубину и скорость
        self.set_yaw(0.0)
        self.set_depth(3.0)
        self.set_speed_forward(0)

        # Список задач для выполнения
        tasks = [self.dive_to_depths] + [self.go_to_do_task] * 5 + [self.go_to_number]

        # Добавляем все задачи в очередь миссий
        for task in tasks:
            self.add_mission(task)

        # Бесконечный цикл выполнения миссий
        while True:
            # Берем миссию из очереди
            mission = self.pop_mission()

            # Выполняем миссию, пока она не завершится успешно
            while not mission():
                self.update()

            # Если все миссии завершены, останавливаем двигатели, останавливаемся, и выходим из бесконечного цикла
            if self.get_missions_length() == 0:
                self.stop_motors()
                self.stop()
                break

    def stop(self):
        # Если речь идет о реальном окружении, освобождаем захваченные камеры
        if self.real:
            self.cap0.release()
            self.cap1.release()
        exit()


robert = Robot()
robert.run()
