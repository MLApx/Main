import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()

from sagemaker.amazon.amazon_estimator import get_image_uri

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")
print (training_image)