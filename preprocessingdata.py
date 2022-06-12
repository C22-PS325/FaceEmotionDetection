"""
Visual_data function presents the number of samples each category has
through a bar plot.
"""
def visual_data(numclass, data):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(numclass, data)
    ax.set(title="Distribution of Dataset's for each Emotion class")
    ax.set(xlabel="Emotions", ylabel="#Images");
    ax.grid()

 
"""
Processing_data function takes a data path and loads all images (along with their
corresponding labels) as numpy arrays per category to the memory.
"""
def processing_data(data_dir):
    subfolder = os.listdir(data_dir)
    print("Proses data...\n")

    image_list=[]
    labels_list = []
    image_per_kelas = []

    for category in subfolder:
        list_image=os.listdir(data_dir +'/'+ category)
        
        print('Loading :', len(list_image), '= images of category class = ', category)
        for img in list_image:
            # Load an image from this path
            pixels=cv2.imread(data_dir + '/'+ category + '/'+ img )
            face_array=cv2.resize(pixels, (224,224), fx=1, fy=1,interpolation = cv2.INTER_CUBIC)
        
            image_list.append(face_array)          
            labels_list.append(category)

        image_per_kelas.append(len(list_image))
    le = LabelEncoder()
    labels = le.fit_transform(labels_list)
    labels = to_categorical(labels, 6)

    visual_data(subfolder, image_per_kelas)

    # Dataset Summary
    print("\nTotal number of uploaded data: ", data.shape[0],", with data shape size,size,channel of picture", (data.shape[1],data.shape[2],data.shape[3]))

    return data, labels