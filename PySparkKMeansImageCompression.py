import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyspark.sql import SparkSession

# Initialize Spark session
def init_spark_session(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

# Read and display an image
def display_image(img_path):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Perform image segmentation using K-Means
def img_seg_with_kmeans(spark_session, img_file, num_clusters):
    img_data, original_dimensions, img_array = read_image(img_file)
    df = convert_to_dataframe(spark_session, img_array)
    centroids_list = initialize_centroids(df, num_clusters)
    for iteration in range(10):
        assigned_clusters = assign_clusters(df, centroids_list)
        centroids_list = recalculate_centroids(assigned_clusters)
    modified_img_array = replace_color(assigned_clusters, centroids_list, original_dimensions)
    new_img_file = save_segmented_image(img_file, num_clusters, modified_img_array)
    return new_img_file

# Read image and prepare data
def read_image(img_file):
    img_data = cv2.imread(img_file)
    original_dimensions = img_data.shape
    img_array = np.asarray(img_data).reshape(-1, 3)
    return img_data, original_dimensions, img_array

# Convert image array to Spark DataFrame
def convert_to_dataframe(spark_session, img_array):
    return spark_session.createDataFrame([tuple(x) for x in img_array.tolist()], ["R", "G", "B"])

# Initialize centroids for K-Means
def initialize_centroids(df, num_clusters):
    centroids_list = df.rdd.takeSample(False, num_clusters)
    return spark_session.sparkContext.broadcast(centroids_list)

# Assign clusters based on centroids
def assign_clusters(df, centroids_list):
    return df.rdd.map(lambda row: calculate_min_distance(row, centroids_list)).toDF(["cluster_id", "coordinate"])

# Calculate minimum distance for cluster assignment
def calculate_min_distance(data_row, centroids_list):
    data_row_array = np.array(list(data_row))
    computed_distances = [np.linalg.norm(data_row_array - np.array(c)) for c in centroids_list.value]
    return (computed_distances.index(min(computed_distances)), list(data_row))

# Recalculate centroids
def recalculate_centroids(assigned_clusters):
    new_centroids_list = assigned_clusters.rdd.map(lambda x: (x[0], (x[1], 1))) \
                              .reduceByKey(lambda x, y: ([x[0][i] + y[0][i] for i in range(len(x[0]))], x[1] + y[1])) \
                              .mapValues(lambda x: [x[0][i] / x[1] for i in range(len(x[0]))]) \
                              .collect()
    new_centroids_list = sorted(new_centroids_list, key=lambda x: x[0])
    new_centroids_list = [x[1] for x in new_centroids_list]
    return spark_session.sparkContext.broadcast(new_centroids_list)

# Replace color based on centroid color
def replace_color(assigned_clusters, centroids_list, original_dimensions):
    color_map = {i: color for i, color in enumerate(centroids_list.value)}
    color_df = assigned_clusters.rdd.map(lambda x: (x[0], color_map[x[0]])).toDF(["cluster_id", "color"])
    modified_img_array = np.array(color_df.select('color').rdd.flatMap(list).collect())
    return modified_img_array.astype(np.uint8).reshape(original_dimensions)

# Save the segmented image
def save_segmented_image(img_file, num_clusters, modified_img_array):
    new_img_file = img_file[:-4] + f"_k={num_clusters}_seg.jpg"
    cv2.imwrite(new_img_file, modified_img_array)
    return new_img_file

# Calculate compression ratio
def calc_compression_ratio(initial_path, final_path):
    original_file_size = os.path.getsize(initial_path)
    compressed_file_size = os.path.getsize(final_path)
    return original_file_size / compressed_file_size

# Main execution function
def main():
    spark_session = init_spark_session("UnsupervisedImgSegment")
    img_dir = "/content/Images"
    cluster_numbers = range(2, 31, 2)
    output_images = {}
    compression_values = {}

    for img_file in os.listdir(img_dir):
        if img_file.endswith(".jpg"):
            full_img_path = os.path.join(img_dir, img_file)
            for num_clusters in cluster_numbers:
                resultant_img_path = img_seg_with_kmeans(spark_session, full_img_path, num_clusters)
                output_images[num_clusters] = resultant_img_path
                compression_val = calc_compression_ratio(full_img_path, resultant_img_path)
                compression_values[num_clusters] = compression_val
                print(f"Compression ratio for {img_file} with k = {num_clusters} is {compression_val}")

    for cluster, img_path in output_images.items():
        print(f"Cluster {cluster}:")
        plt.figure(figsize=(10,10))
        if os.path.isfile(img_path):
            display_image(img_path)
        else:
            print(f"File does not exist: {img_path}")

if __name__ == "__main__":
    main()
