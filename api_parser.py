import requests
import os
import numpy as np
import argparse
import cv2
import json
import shutil


def get_class_name(image_path):
    """
    call fast api to get the image class
    Args:
        image_path: path to ct scan image
    Returns:
        json : API json response
    """
    with open(image_path, "rb") as f:
        r = requests.post(
            "http://127.0.0.1:8000/get_image_class",
            file_names={"file_name": (os.path.basename(image_path), f)},
        )
    print(r.text)
    out = json.loads(r.text)

    return out


def get_model_metrics(covid_images_list, non_covid_images_list, images_path):

    """
    call fast api to get model metrics

    Args:
        covid_images_list: list of images belonging to class "Covid",
        non_covid_images_list: list of images belonging to class "Non_Covid"
        images_path: Path to the base folder which has both the images folder
        Note: The base folder should have two folders, saved as "COVID" and "NON_COVID" with their corresponding images

    Returns:
        json : API json response which has the model metrics in a dictionary format
    """
    file_name_list = []

    for file_name in covid_images_list:
        file_name_list.append(
            (
                "covid_files",
                (
                    file_name,
                    open(os.path.join(images_path, "COVID", file_name), "rb"),
                    "image/png",
                ),
            )
        )

    for file_name in non_covid_images_list:
        file_name_list.append(
            (
                "non_covid_files",
                (
                    file_name,
                    open(os.path.join(images_path, "NON_COVID", file_name), "rb"),
                    "image/png",
                ),
            )
        )

    r = requests.post(
        "http://127.0.0.1:8000/get_model_metrics",
        files=file_name_list,
    )
    # print(r.text)

    out = json.loads(r.text)

    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--choice",
        help="a: For classifying an image/images, b: For looking at model metrics",
    )

    args = parser.parse_args()

    choice = args.choice

    if choice == "a":

        input_path = input("Please provide the path to the image file_name or dir: ")
        os.makedirs("./Custom/Covid", exist_ok=True)
        os.makedirs("./Custom/non-Covid", exist_ok=True)

        if os.path.isfile_name(input_path):
            print("Found the file_name passed")

            if (
                input_path.endswith(".jpg")
                or input_path.endswith(".png")
                or input_path.endswith(".jpeg")
            ):

                name = input_path.split("/")[-1]

                class_name = get_class_name(input_path)
                print(class_name)
                if class_name["Class"] == "Covid":
                    shutil.copyfile_name(
                        input_path, os.path.join("./Custom/Covid", name)
                    )

                else:
                    shutil.copyfile_name(
                        input_path, os.path.join("./Custom/non-Covid", name)
                    )

                print("Image saved")

            else:
                print("Invalid Image type")

        elif os.path.isdir(input_path):
            print("Found the directory passed")

            file_names = os.listdir(input_path)

            for file_name in file_names:
                file_name = file_name.lower()
                if (
                    file_name.endswith(".jpg")
                    or file_name.endswith(".png")
                    or file_name.endswith(".jpeg")
                ):
                    print(file_name)
                    class_name = get_class_name(os.path.join(input_path, file_name))
                    print(class_name)
                    if class_name["Class"] == "Covid":
                        shutil.copyfile_name(
                            os.path.join(input_path, file_name),
                            os.path.join("./Custom/Covid", file_name),
                        )

                    else:
                        shutil.copyfile_name(
                            os.path.join(input_path, file_name),
                            os.path.join("./Custom/non-Covid", file_name),
                        )

                    print("Image saved")
                    print("----")

                else:
                    print("Incorrect file_name type!")

    if choice == "b":
        data_path = input("Please enter the path of the directroy: ")

        covid_images = os.listdir(os.path.join(data_path, "COVID"))
        non_covid_images = os.listdir(os.path.join(data_path, "NON_COVID"))

        print("dir extracted successfully")
        print("Evaluation may take a few minutes if no GPU and the dataset is big!")
        print("\n")
        metrics = get_model_metrics(covid_images, non_covid_images, data_path)
        print("Model Metrics: ")
        print("\n")
        print(f"Accuracy: {metrics['Model_accuracy']}")
        print(f"Precision: {metrics['Model Precision']}")
        print(f"Recall: {metrics['Model Recall']}")
        print(f"F1 Score: {metrics['Model F1 score']}")

        print(f"List of files classified incorrectly as Non_Covid:")
        for name in metrics["List of false Positives"]:
            print(name)

        print(f"List of files classified incorrectly as Covid:")
        for name in metrics["List of False Negatives"]:
            print(name)

        print("success")


if __name__ == "__main__":
    main()
