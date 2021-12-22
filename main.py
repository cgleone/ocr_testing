import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
import time
import preprocessing as pre
from datetime import datetime
import accuracy_testing as accuracy
from PIL import Image

def get_text(file, test_value_dict):

    txt_file = create_output_file(file)
    path = 'prepped_pics/' + file

    img = cv2.imread(path)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    img = pre.deskew(img) # always leave on


    img = pre.greyscale(img) # always leave on
   # img = pre.rescale(img, test_value_dict.get_value('rescale'))
    img = pre.rescale(img, 1.5)
    # img = pre.rescale(img, test_value_dict.get_value('rescale'))

    # NOW MAKE IT A PILLOW

    pillow_path = save_for_pillowing(img, file)
    pil_img = get_pil_img(pillow_path)


    pil_img = pre.sharpen(pil_img, test_value_dict.get_value('sharpen'))
    pil_img = pre.brighten(pil_img, test_value_dict.get_value('brighten'))
    pil_img = pre.contrast(pil_img, test_value_dict.get_value('contrast'))

    pil_img = pre.sharpen(pil_img, 3)
    pil_img = pre.brighten(pil_img, 1.5)
    pil_img = pre.contrast(pil_img, 2.5)

    # BACK TO OPENCV NOW
    # img = np.array(pil_img)  # alrighty done it's a opencv now

    img = pre.thresholding(img) # always leave on

   # img = pre.closing(img, test_value_dict.get_value('closing'))
    img = pre.gaussian_blur(img, test_value_dict.get_value('gaussian'))
   # img = pre.gaussian_blur(img, 1)
    # img = pre.gaussian_blur(img, test_value_dict.get_value('gaussian'))
    img = pre.median_blur(img, test_value_dict.get_value('median'))

    # img = pre.closing(img, test_value_dict.get_value('closing'))

    # img = pre.gaussian_blur(img, test_value_dict.get_value('gaussian'))
    #cv2.imwrite('Modeling_graphs/9x9 Gaussian pdf.jpg', img)
    img = pre.erode(img, test_value_dict.get_value('erosion'))
    save_preprocessing_img(img, test_value_dict)
    # img = pre.dilate(img, test_value_dict.get_value('dilation'))


    # img = pre.median_blur(img, test_value_dict.get_value('median'))
    # img = pre.averaging_blur(img, test_value_dict.get_value('averaging'))

    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    #img = pre.canny(img)
    #img = pre.erode(img)
    #img = pre.dilate(img)


    converted = pytesseract.image_to_string(img)
    print("Text found in '{}'\n".format(file))
    print(converted)

    accuracies = accuracy.get_accuracies(file, converted)
    add_to_file(txt_file, converted, pre.method_list, accuracies)
    print(accuracies)

    return accuracies

def save_preprocessing_img(img, test_value_dict):
    iter_value = str(test_value_dict.get_value('erosion'))
    pic_name = 'Test_output_pics/TestMedian3Erosion' + iter_value + '.jpg'
    cv2.imwrite(pic_name, img)


def convert_pdf(path):
    new_name = get_new_name(path)
    image = convert_from_path(path, paths_only=True, output_folder='converted_jpgs', fmt='jpeg', output_file=new_name)
    return image[0]


def save_for_pillowing(img, filename):
    path = 'pics_for_pillowing/' + filename
    cv2.imwrite(path, img)
    return path


def get_pil_img(path):
    return Image.open(path)


def get_new_name(path):
    split_path = path.split('/')
    pdf_name = split_path[len(split_path)-1]
    new_name = pdf_name.rstrip('.pdf')
    return new_name


def create_output_file(img_name):
    txt_name = img_name.split('.')[0]
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt = now.strftime("_%d_%m_%Y_%H-%M-%S")
    file = open('conversion_reports/' + txt_name + dt + '.txt', "w+")
    return file


def add_to_file(file, body_text=0, processing_methods=0, accuracies=0):
    if body_text:
        # write the body
        file.write(body_text)
        file.write("\n\n----------------------------------TEXT COMPLETE----------------------------------------\n\n")
    if processing_methods:
        # write the processing methods
        file.write("The pre-processing methods used in this OCR text conversion were:\n")
        for method in processing_methods:
            file.write(method + '\n')
    else:
        file.write("No pre-processing methods were used to produce this output.\n")

    if accuracies:
        file.write("\nThe accuracy of this OCR reading is shown below:\n\n")
        file.write("SequenceMatcher accuracy ratio (converted:original) = {}\n".format(accuracies[0]))
        file.write("Levenshtein accuracy ratio (converted:original) = {}\n".format(accuracies[1]))

    if not body_text and not processing_methods and not accuracies:
        print("add_to_file function was called but not passed anything to add. /n")

    return file


def prep_image(file, noisify=False):

    path = 'pics/' + file
    delete_after = False
    new_name = file
    if file.endswith('.pdf'):
        path = convert_pdf(path)
        new_name = file.rstrip('.pdf') + '.jpg'
        delete_after = True

    if noisify:
        with Image.open(path) as img:
            result = accuracy.noisify(img, rotation=72, brightness=1, contrast=1, sharpness=1)
            # result.show()
    else:
        result = Image.open(path)

    result = result.convert("RGB")
    result.save('prepped_pics/' +new_name)

    if delete_after:
        os.remove(path)  # get rid of the extra image file once we're done with it, if we made one

    return new_name


if __name__ == '__main__':


    file = '12486 4 anon.pdf'
    #file = 'report_body_1.jpg'
    #file = 'Turnbull_2.pdf'
    new_file = prep_image(file, noisify=False)



    start_time = time.time()

    get_text(new_file, accuracy.TestValueDict())


    accuracy.optimize_single_method(new_file)

    executionTime = (time.time() - start_time)
    print("Execution Time: {} seconds".format(round(executionTime, 2)))