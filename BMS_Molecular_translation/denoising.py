#Ref:https://www.kaggle.com/maksymshkliarevskyi/bms-mol-tr-approaches-eda-denoise-baseline

def image_denoising(img_path, dot_size = 2):
    """
    Source: https://stackoverflow.com/questions/48681465/how-do-i-remove-the-dots-noise-without-damaging-the-text
    Function for removing noise in the form of small dots. 
    The input takes the path to the image.
    Increase 'dot_size' parameter to increase the size of the areas (dots) to be removed
    """
    image = io.imread(img_path)
    _, BW = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(BW, None, None, None, 
                                         8, cv2.CV_32S)
    sizes = stats[1:, -1]
    image2 = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= dot_size: 
            image2[labels == i + 1] = 255
    image = cv2.bitwise_not(image2)
    return image



#other version
#Ref: https://www.kaggle.com/paulorzp/denoise-images
def visualize_image_denoise(image_id):
    plt.figure(figsize=(10, 8))  
    image = cv2.imread(convert_image_id_2_path(image_id), cv2.IMREAD_GRAYSCALE)
    _, blackAndWhite = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= 2:   #filter small dotted regions
            img2[labels == i + 1] = 255
    image = cv2.bitwise_not(img2)
    plt.imshow(image)    
    plt.axis("off")
    plt.show()

i=0
visualize_image_denoise(df_labels.index[i])