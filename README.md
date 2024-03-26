
# Unsupervised Image Compression Using K-Means with Spark

This repository contains a Python script for performing unsupervised image segmentation using K-Means clustering algorithm, implemented in Apache Spark. The script processes images by segmenting them into different clusters based on pixel color, and recomposing the image with centroid colors of these clusters.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
- [Example](#example)
- [Authors](#authors)

## Requirements

This script requires the following:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Apache Spark (PySpark)

## Installation

To set up your environment to run this script, follow these steps:

1. **Python Installation**: Ensure you have Python 3.x installed on your machine. 

2. **Dependency Installation**: Install the required libraries using pip:

    ```bash
    pip install numpy opencv-python matplotlib pyspark
    ```

3. **Apache Spark Setup**: Make sure Apache Spark is installed and properly set up on your system. [Apache Spark's official site](https://spark.apache.org/downloads.html) provides detailed instructions.

## Usage

The script is designed to be executed from the command line. Ensure all the required files and images are in the correct directories before running the script.

## Functionality

The script includes the following key functions:

- `init_spark_session(app_name)`: Initializes a Spark session for distributed computing.

- `display_image(img_path)`: Uses Matplotlib to display an image from the given file path.

- `img_seg_with_kmeans(spark_session, img_file, num_clusters)`: The main function that performs image segmentation using K-Means. It takes a Spark session, the path to an image file, and the desired number of clusters as inputs.

- `read_image(img_file)`: Utilizes OpenCV to read an image file and converts it into a format suitable for processing.

- `convert_to_dataframe(spark_session, img_array)`: Converts the image data into a Spark DataFrame for distributed processing.

- `initialize_centroids(...)`, `assign_clusters(...)`, `calculate_min_distance(...)`, `recalculate_centroids(...)`, `replace_color(...)`: These functions are utilized within `img_seg_with_kmeans` to implement the K-Means clustering algorithm.

- `save_segmented_image(...)`: Saves the output of the segmented image.

- `calc_compression_ratio(...)`: Calculates and returns the compression ratio of the segmented image compared to the original.

- `main()`: The main function which orchestrates the segmentation process across multiple images and cluster sizes.

## Results

<img width="495" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/c1b0b73d-1542-4128-8305-7c11ccf23909">
<img width="425" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/dea54198-ae55-406b-a893-1be90f72136b">

<img width="476" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/ca97e80c-1df2-4e2b-b62b-5312a92c21a1">

<img width="599" alt="Screenshot 2024-03-25 at 7 18 43 PM" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/d084d0ba-e0d4-411c-ae15-1cee272a29e1">

<img width="449" alt="image" src="https://github.com/nishan2293/PySparkKMeansImageCompression/assets/157925518/bb4db431-9798-4746-acfc-bdd83107cd38">



## Example

To run the script, use the following command:

```python
file_name.py
```

## Authors

- [@nishan2293](https://github.com/nishan2293)

