class Camera:
    def __init__(self, rotation, translation, image):
        self.quaternion = rotation
        self.translation = translation
        self.image = image