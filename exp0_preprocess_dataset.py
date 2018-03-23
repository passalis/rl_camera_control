import cv2
from os import listdir
from os.path import isfile, join
import pickle


def preprocess_dataset(dataset_path='data/dataset', dataset_pickle_path='data/dataset.pickle'):
    """
    Preprocesses the HP Database to extract the pose annotations
    :param dataset_path:
    :return:
    """

    # Load dataset
    persons = []
    for person in range(15):
        pose_dictionary1 = {}
        pose_dictionary2 = {}

        if person + 1 < 10:
            person_id = '0' + str(person + 1)
        else:
            person_id = str(person + 1)

        cur_file_path = join(dataset_path, 'Person' + person_id)

        person_files = [f for f in listdir(cur_file_path) if isfile(join(cur_file_path, f)) and f[-3:] == 'jpg']

        for cur_file in person_files:

            shot_id = int(cur_file[8])

            # Load annotation
            annotation_txt = cur_file[:-3] + 'txt'
            with open(join(cur_file_path, annotation_txt)) as f:
                data = f.readlines()

            # Extract bounding box
            x_center, y_center, width, height = [int(x[:-1]) for x in data[3:]]

            # Extract pose
            pose_info = cur_file[11:-4]
            if pose_info[1] == '0':
                tilt = 0
                pan = int(pose_info[2:])
            else:
                tilt = int(pose_info[:3])
                pan = int(pose_info[3:])

            cur_img_original = cv2.imread(join(cur_file_path, cur_file))

            width, height = int(width / 2), int(height / 2)
            lower_y = max(y_center - height, 0)
            lower_x = max(x_center - width, 0)
            upper_y = min(y_center + height, cur_img_original.shape[0])
            upper_x = min(x_center + width, cur_img_original.shape[1])

            cur_img = cur_img_original[lower_y:upper_y, lower_x:upper_x:]
            cur_img = cv2.resize(cur_img, (64, 64))

            x_scale, y_scale = 256.0/cur_img_original.shape[1], 256.0/cur_img_original.shape[0]

            cur_img_original = cv2.resize(cur_img_original, (256, 256))

            face_position = (lower_x*x_scale, lower_y*y_scale, upper_x*x_scale, upper_y*y_scale)
            face_position = [int(x) for x in face_position]


            if shot_id == 1:
                pose_dictionary1[(tilt, pan)] = (cur_img, cur_img_original, face_position)

            if shot_id == 2:
                pose_dictionary2[(tilt, pan)] = (cur_img, cur_img_original, face_position)

        persons.append(pose_dictionary1)
        persons.append(pose_dictionary2)

    with open(dataset_pickle_path, 'wb') as f:
        pickle.dump(persons, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # You must download the HPID (http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html) to 'data/dataset'

    print("Preprocessing dataset (this should take a minute) ...")
    preprocess_dataset()
