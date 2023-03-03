import cv2
import numpy as np

objects = {}
video = cv2.VideoCapture(0)


def match_template(method):
    while True:
        cv2.namedWindow("Live video", cv2.WINDOW_NORMAL)
        ok, image_template1 = video.read()
        cv2.imshow("Live video", image_template1)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            for i in range(3):
                box = cv2.selectROI("Live video", image_template1)
                user_obj = str(input("Enter the name of the object: "))
                objects[i] = user_obj
                point1 = (int(box[0]), int(box[1]))
                point2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                image_template1 = image_template1[point1[1]:point2[1], point1[0]:point2[0]]
                #image_template1_gray = cv2.cvtColor(image_template1, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("Template " + str(i) + ".jpg", image_template1)
                ok, image_template1 = video.read()
        elif key == ord('r'):
            if len(objects) == 0:
                print('Capture objects first')
                continue
            template0 = cv2.imread("Template 0.jpg")
            template1 = cv2.imread("Template 1.jpg")
            template2 = cv2.imread("Template 2.jpg")
            width_temp0, height_temp0 = template0.shape[:2][::-1]
            width_temp1, height_temp1 = template1.shape[:2][::-1]
            width_temp2, height_temp2 = template2.shape[:2][::-1]
            method_ = eval(method)
            i = 0
            while True:
                ok, image_template1 = video.read()
                #image_template1_gray = cv2.cvtColor(image_template1, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(image_template1, template0, method_)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                cv2.rectangle(image_template1, top_left, (top_left[0] + width_temp0, top_left[1] + height_temp0),
                              (0, 0,
                               255), 2)
                cv2.putText(image_template1, objects[0], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(image_template1, method, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                            2)

                result = cv2.matchTemplate(image_template1, template1, method_)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + width_temp1, top_left[1] + height_temp1)
                cv2.rectangle(image_template1, top_left, bottom_right, (0, 0, 255), 2)
                cv2.putText(image_template1, objects[1], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                result = cv2.matchTemplate(image_template1, template2, method_)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + width_temp2, top_left[1] + height_temp2)
                cv2.rectangle(image_template1, top_left, bottom_right, (0, 0, 255), 2)
                cv2.putText(image_template1, objects[2], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(image_template1, str(i), (top_left[0], top_left[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Live video", image_template1)
                ok, image_template1 = video.read()
                key = cv2.waitKey(-1)
                if key == ord('e'):
                    break
                if key == ord('p'):
                    cv2.waitKey(-1)
                i += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = int(input(
        "Select a method" + '\n' + "1] TM_CCOEFF" + '\n' + "2] TM_CCOEFF_NORMED" + '\n' + "3] TM_CCORR" +
        '\n' + "4]TM_CCORR_NORMED" + '\n' + '5]TM_SQDIFF' + '\n' + '6] TM_SQDIFF_NORMED'))
    if choice == 1:
        match_template("cv2.TM_CCOEFF")
    elif choice == 2:
        match_template("cv2.TM_CCOEFF_NORMED")
    elif choice == 3:
        match_template("cv2.TM_CCORR")
    elif choice == 4:
        match_template("cv2.TM_CCORR_NORMED")
    elif choice == 5:
        match_template("cv2.TM_SQDIFF")
    elif choice == 6:
        match_template("cv2.TM_SQDIFF_NORMED")

