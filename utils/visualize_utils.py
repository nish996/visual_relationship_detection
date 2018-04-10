import numpy as np
import cv2



'''
function for visualizing bounding boxes and labels with score
given bounding boxes of the image
'''
def visualize_pred(img_path,bboxes):
    # load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    #print(bboxes)
    labels_to_names = labelsToNames()
    for i in range(bboxes.shape[0]):
        b = bboxes[i,0:4].astype(int)
        label = bboxes[i,4]
        score = bboxes[i,5]
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

'''
function for visualizing bounding boxes given bounding boxes of the image
'''
def  visualize_bboxes(img_path,bboxes):
    # load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    #print(bboxes)
    labels_to_names = labelsToNames()
    for i in range(bboxes.shape[0]):
        b = bboxes[i,0:4].astype(int)
        label = bboxes[i,4]
        score = bboxes[i,5]
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()



def main():
    x  = 10

if __name__ == '__main__':
    main()