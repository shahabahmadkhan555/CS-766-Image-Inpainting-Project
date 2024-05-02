import cv2
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from ultralytics import YOLO

class Segmentor():
    def __init__(self, input_image_path, objects=[0], output_dir='', save_images=True):
        self.model = YOLO('yolov8m-seg.pt')
        self.input_image_path = input_image_path
        self.image = np.array(Image.open(input_image_path))
        self.output_dir = output_dir    # TODO
        self.save_images = save_images  # TODO
        self.OBJECTS = objects          # Objects to remove (defaults to [0] = people)

    def get_results(self, input_file_path):
        return self.model(input_file_path)

    def save_image(self, img, output_path='/', ):
        cv2.imwrite(output_path, img.cpu().numpy())

    def display_image(self, img, title=''):
        plt.figure()
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, interpolation='none')
        plt.show()

    def display_mask_overlay(self, img, mask, title=''):
        plt.figure()
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, 'gray', interpolation='none')
        mask = mask.cpu().numpy()
        mask = np.ma.masked_where(mask == 0, mask)
        plt.imshow(mask, 'jet', interpolation='none', alpha=0.5)
        plt.show()

    def display_and_save_mask_overlay(self, img, mask, title='', output_path='/'):
        plt.figure()
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, 'gray', interpolation='none')
        mask = mask.cpu().numpy()
        mask = np.ma.masked_where(mask == 0, mask)
        plt.imshow(mask, 'jet', interpolation='none', alpha=0.5)
        plt.savefig(output_path)
        plt.show()

    def display_and_save_image(self, img, title='', output_path='/'):
        plt.figure()
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, interpolation='none')
        plt.savefig(output_path)
        plt.show()

    def run_detection_segmentation(self):
        self.display_image(self.image, title="Original Image")

        results = self.get_results(self.input_image_path)
        result = results[0] # since we only pass a single image to segment, there is only one result

        pred_array = result.plot()  # BGR numpy array of predictions
        pred_img = Image.fromarray(pred_array[..., ::-1])  # Convert to RGB PIL image

        # self.display_and_save_image(pred_img, output_path='predictions.jpg')
        self.display_image(pred_img, title="Predictions")

        # Get the lists of masks and bounding boxes:
        if result.masks is None:
            return None, None, None
        masks = result.masks.data
        boxes = result.boxes.data

        # Save the masks and boxes:
        self.masks = masks
        self.boxes = boxes
        self.detected_objects = boxes[:, 5]

        detected_objects = boxes[:, 5]

        # Extract a mask with all detected objects:
        obj_indices = torch.where(detected_objects != -1)
        obj_masks = masks[obj_indices]
        obj_mask = torch.any(obj_masks, dim=0).int() * 255
        self.save_image(obj_mask, 'all-detected-objects-masks.jpg')

        return masks, detected_objects, pred_img

    # Get the mask that includes specified objects:
    def get_mask(self, objects=None):
        if objects is None:
            objects = self.OBJECTS

        masks, detected_objects, predictions_img = self.run_detection_segmentation()
        if masks is None:
            return None

        # Extract a single mask that contains all segmentations of specified object types:
        object_indices = []

        # Mask for all instances of an object type:
        for id in objects:

            obj_indices = torch.where(detected_objects == id)
            object_indices.append(obj_indices[0])
            obj_masks = masks[obj_indices]
            obj_mask = torch.any(obj_masks, dim=0).int() * 255 # Tensor

            #self.save_image(obj_mask, str(f'object_class{id}_mask.jpg'))

            # Resize mask to image size:
            # image_height, image_width = img.shape[:2]
            # obj_mask = cv2.resize(np.array(obj_mask, dtype='uint8'), (image_width, image_height), interpolation=cv2.INTER_CUBIC)
            
            # OR Resize image to mask size:
            mask_height, mask_width = obj_mask.shape[:2]
            resized_img = cv2.resize(self.image, (mask_width, mask_height), interpolation=cv2.INTER_CUBIC)

            # Convert into a logical mask that can be directly applied to an image:
            obj_mask = obj_mask.cpu().numpy() 
            actual_mask = np.ma.masked_where(obj_mask == 0, obj_mask)

            # Plot the input image, the mask overlayed on the image, and the image after the mask is applied:
            fig, ax = plt.subplots(nrows=1, ncols=3, tight_layout=True)
            ax[0].set_title("Original Image")
            ax[0].axis('off')
            ax[0].imshow(self.image, 'gray', interpolation='none')

            # Mask overlay:
            ax[1].set_title(str(f'{id} Mask Overlay'))
            ax[1].axis('off')      
            ax[1].imshow(resized_img, 'gray', interpolation='none')
            ax[1].imshow(obj_mask, 'jet', interpolation='none', alpha=0.5)

            # Mask applied to image:
            ax[2].set_title(str(f'{id} Mask Directly Applied'))
            ax[2].axis('off')
            ax[2].imshow(resized_img, 'gray', interpolation='none')
            ax[2].imshow(actual_mask, 'jet', interpolation='none', alpha=0.5)
            # plt.savefig(str(f'object_class{id}_applied_mask.jpg'), bbox_inches='tight', pad_inches = 0)
            fig.show()

        # Combine masks of specified object types to extract a single mask with all wanted segmentations:
        object_indices = torch.cat(object_indices, dim=0)
        object_masks = masks[object_indices]
        object_mask = torch.any(object_masks, dim=0).int() * 255
        self.save_image(object_mask, 'objects-to-remove-masks.jpg')

        self.display_image(object_mask.cpu().numpy(), title="Mask of Objects to Remove")
        self.display_and_save_mask_overlay(resized_img, object_mask, title="Final Mask Applied to Image", output_path='object_removal_segmentation.jpg')

        # Plot the input image, the object detection predictions, and the extracted mask:
        fig, ax = plt.subplots(nrows=1, ncols=3, tight_layout=True)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[0].imshow(self.image, cmap='gray', interpolation='none')

        # Mask overlay:
        ax[1].set_title("Predictions")
        ax[1].axis('off')
        ax[1].imshow(predictions_img, cmap='gray', interpolation='none')

        # Mask applied to image:
        ax[2].set_title("Masks")
        ax[2].axis('off')
        ax[2].imshow(object_mask.cpu().numpy(), cmap='gray', interpolation='none')
        plt.savefig(str(f'predictions_and_mask.jpg'), bbox_inches='tight', pad_inches = 0)
        fig.show()

        return object_mask
