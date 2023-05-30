def patch_coverage(mask):
    """_summary_
    function to get auxiliary information of image
    Args:
        mask_path (string): mask image's path
    Returns:
        dict: a dict of description about realtive area of mask in image and its spread across four quadrants
    """
    mask = mask
    h, w = mask.shape
    total_pix = np.shape(mask)[0] * np.shape(mask)[1]
    num_white_pix = np.sum(mask == 1)
    coverage = num_white_pix / total_pix

    g_one_white_pix = np.sum(mask[0 : int(h / 3), 0 : int(w / 3)] == 1)
    g_two_white_pix = np.sum(mask[0 : int(h / 3), int(w / 3) : int(2 * w / 3)] == 1)
    g_three_white_pix = np.sum(mask[0 : int(h / 3), int(2 * w / 3) :] == 1)
    g_four_white_pix = np.sum(mask[int(h / 3) : int(2 * h / 3), 0 : int(w / 3)] == 1)
    g_five_white_pix = np.sum(
        mask[int(h / 3) : int(2 * h / 3), int(w / 3) : int(2 * w / 3)] == 1
    )
    g_six_white_pix = np.sum(mask[int(h / 3) : int(2 * h / 3), int(2 * w / 3) :] == 1)
    g_seven_white_pix = np.sum(mask[int(2 * h / 3) :, 0 : int(w / 3)] == 1)
    g_eight_white_pix = np.sum(mask[int(2 * h / 3) :, int(w / 3) : int(2 * w / 3)] == 1)
    g_nine_white_pix = np.sum(mask[int(2 * h / 3) :, int(2 * w / 3) :] == 1)

    one_per = g_one_white_pix / num_white_pix
    two_per = g_two_white_pix / num_white_pix
    three_per = g_three_white_pix / num_white_pix
    four_per = g_four_white_pix / num_white_pix
    five_per = g_five_white_pix / num_white_pix
    six_per = g_six_white_pix / num_white_pix
    seven_per = g_seven_white_pix / num_white_pix
    eight_per = g_eight_white_pix / num_white_pix
    nine_per = g_nine_white_pix / num_white_pix

    return {
        "coverage": coverage,
        "per_grid": {
            "top left": one_per,
            "top": two_per,
            "top right": three_per,
            "left": four_per,
            "center": five_per,
            "right": six_per,
            "bottom left": seven_per,
            "bottom": eight_per,
            "bottom right": nine_per,
        },
    }


def get_mask_decription(mask_ori):
    """_summary_
    function to get auxiliary information of image
    Args:
        mask_ori (array): 2D numpy array of mask image
    Returns:
        _type_: _description_
    """
    # mask_ori = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = np.array((mask_ori / 255.0) > 0.5, dtype=int)

    # get masks labelled with different values
    label_im, nb_labels = ndimage.label(mask)
    res = {}

    j = 0
    sizes = []
    positions = []

    multiple = False
    num = nb_labels
    for i in range(nb_labels):

        if i > 0:
            multiple = True
        mask_compare = np.full(np.shape(label_im), i + 1)
        # check equality test and have the value 1 on the location of each mask
        separate_mask = np.equal(label_im, mask_compare).astype(int) * mask_ori

        # separate_mask_img = Image.fromarray((separate_mask * 255).astype(np.uint8))

        res_i = patch_coverage(separate_mask / 255.0)
        if res_i["coverage"] > 0.0001:
            res[i] = res_i
            j = j + 1
            if res_i["coverage"] > 0.30:
                sizes.append("large")
            elif res_i["coverage"] > 0.08:
                sizes.append("medium")
            elif res_i["coverage"] < 0.001:
                sizes.append("tiny")
            else:
                sizes.append("small")
        pos = res_i["per_grid"]
        posnonzero = {k: v for k, v in pos.items() if v != 0}
        pos_str = ""

        if len(posnonzero) == 0:
            pass
        elif len(posnonzero) == 1:
            pos_str = list(posnonzero.keys())[0]
        elif len(posnonzero) > 1:
            pos_signigicant = {k: v for k, v in pos.items() if v >= 0.1}

            if len(pos_signigicant) == 0:
                posnonzero_sorted = dict(
                    sorted(posnonzero.items(), key=lambda item: item[1])
                )
                pos_str = list(posnonzero_sorted.keys())[0]

            elif len(pos_signigicant) == 1:
                pos_str = list(pos_signigicant.keys())[0]

            elif len(pos_signigicant) > 1:
                bbox = mask_to_bbox(separate_mask)[0]

                (c_x, c_y) = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)

                h, w = mask.shape
                if c_x < h / 3:
                    p_x = "top"
                elif c_x >= h / 3 and c_x <= 2 * h / 3:
                    p_x = "center"
                elif c_x > 2 * h / 3:
                    p_x = "bottom"

                if c_y < h / 3:
                    p_y = "left"
                elif c_y >= h / 3 and c_y <= 2 * h / 3:
                    p_y = "center"
                elif c_y > 2 * h / 3:
                    p_y = "right"

                if p_x == "center":
                    pos_str = p_y
                elif p_y == "center":
                    pos_str = p_x
                else:
                    pos_str = p_x + " " + p_y
        positions.append(pos_str)

    tiny_obj = sizes.count("tiny")
    small_obj = sizes.count("small")
    medium_obj = sizes.count("medium")
    large_obj = sizes.count("large")

    sizes = []

    if tiny_obj > 0:
        sizes.append("tiny")
    if small_obj > 0:
        sizes.append("small")
    if medium_obj > 0:
        sizes.append("medium")
    if large_obj > 0:
        sizes.append("large")

    return ", ".join(sizes), ", ".join(positions), num
