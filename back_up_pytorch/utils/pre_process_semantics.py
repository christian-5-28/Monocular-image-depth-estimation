import os
from PIL import Image


def collect_semantic_labels_v_kitti(path):
    """
    collects the pixel values for a specific
    directory of the virtual Kitti dataset,
    avoiding repetition due to instance segmentation
    for specific classes
    """

    car_founded = False
    van_founded = False
    labels = {}
    with open(path) as data:
            next(data)
            for line in data:
                if car_founded and van_founded:
                    return labels
                line = line.replace(":", " ")
                line = line.split()
                if "Car" in line and not car_founded:
                    labels['car'] = (int(line[-3]), int(line[-2]), int(line[-1]))
                    car_founded = True
                elif "Van" in line and not van_founded:
                    labels['van'] = (int(line[-3]), int(line[-2]), int(line[-1]))
                    van_founded = True
                elif line[0] != "Car" and line[0] != "Van":
                    labels[line[0].lower()] = (int(line[-3]), int(line[-2]), int(line[-1]))

    return labels


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def image_paths(path):
    """
    traverse root directory, and list
    directories as dirs and files as files
    """
    images_paths = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            root_list = splitall(root)
            file_path = os.path.join(root, file)
            images_paths[file_path] = root_list[-2]
    return images_paths


def get_instances_pixel_values(label, path):
    """
    getting pixel values for each instance
    of a specific class in a specific directory

    :param label: chosen class
    :param path: chosen directory
    :return:
    """

    pixel_values = []
    with open(path) as data:
            next(data)
            for line in data:
                line = line.replace(":", " ")
                line = line.split()
                if label in line:
                    pixel_values.append((int(line[-3]), int(line[-2]), int(line[-1])))

    return pixel_values


def instance_to_semantic(root_path, sequences):
    """
    converts in semantic segmentation all the
    classes used also for instance segmentation
    :param root_path:
    :param sequences:
    :return:
    """

    image_paths_list = image_paths(root_path)

    pixel_cars = {}
    pixel_vans = {}

    for seq in sequences:
        pixel_values_car = get_instances_pixel_values("Car", sequences[seq])
        pixel_values_van = get_instances_pixel_values("Van", sequences[seq])

        # saving all the instance pixel values for car and van of the specific sequence
        pixel_cars[seq] = pixel_values_car
        pixel_vans[seq] = pixel_values_van

    for index, path in enumerate(image_paths_list):

        sequence = image_paths_list[path]

        # selecting the instance pixel values of the specific sequence
        pixel_car_values = pixel_cars[sequence]
        pixel_van_values = pixel_vans[sequence]

        old_image = Image.open(path)

        # loading the pixel values of the current image
        pixels = old_image.load()

        for y in range(old_image.size[1]):
            for x in range(old_image.size[0]):

                # check if pixel values in the instances values
                if tuple(pixels[x, y]) in pixel_car_values[1:]:

                    # in the positive case, we change the values as the first instance pixel value found
                    pixels[x, y] = pixel_car_values[0]

                if tuple(pixels[x, y]) in pixel_van_values[1:]:
                    pixels[x, y] = pixel_van_values[0]



        print("saving image: " + path)
        print("images left: %d of %d" % (index, len(image_paths_list)))

        old_image.save(path)


def mapping_vkitti_to_cityscapes(root_path, sequences, label_map_vkitti_city_scapes):
    """
    creates label images using as id the cityScapes class ids
    """

    image_paths_list = image_paths(root_path)
    labels_sequences = {}

    # getting pixel values for each class of each sequence
    for seq in sequences:
        labels_sequences[seq] = collect_semantic_labels_v_kitti(sequences[seq])

    rgb_dict = {}
    current_seq = ''
    for index, path in enumerate(image_paths_list):
        if index == 0:
            current_seq = image_paths_list[path]

        sequence = image_paths_list[path]

        # creating dictionary having as key the rgb value
        # and as value the class label
        if current_seq != sequence or index == 0:
            current_seq = sequence
            label_seq = labels_sequences[current_seq]
            rgb_dict = {}
            for label, rgb in label_seq.items():
                rgb_dict[rgb] = label

        old_image = Image.open(path)
        width, height = old_image.size
        new_image = Image.new(mode='L', size=(width, height))
        old_pixels = old_image.load()
        new_pixels = new_image.load()

        for y in range(old_image.size[1]):
            for x in range(old_image.size[0]):

                rgb_key = tuple(old_pixels[x, y])

                # getting the vkitti label
                vkitti_label = rgb_dict[rgb_key]

                # getting cityscapes label id using the map between cirtScapes and virtual Kitti
                cityscapes_label_id = label_map_vkitti_city_scapes[vkitti_label]

                # saving in the new image the label id in the pixel value
                new_pixels[x, y] = cityscapes_label_id

        print("saving image: " + path)
        print("images left: %d of %d" % (index, len(image_paths_list)))

        new_image.save(path)
