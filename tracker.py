import math

class Tracker:
    def __init__(self):
        # Luu center point obj
        self.center_points = {}
        # tang id khi co obj moi
        self.id_count = 1

    def update(self, objects_rect):
        # boxes and ids
        objects_bbs_ids = []

        # Lay diem trung tam new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # check doi tuong da được phát hiện
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                # khoảng cách giữa tâm của bounding box hiện tại và tâm của đối tượng đã được theo dõi trước đó
                if dist < 200:
                    self.center_points[id] = (cx, cy)
                    # cập nhật đối tượng id trước đó với center point của bb hiện tại 
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # gán ID cho đối tượng mới 
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
