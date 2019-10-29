import os
import sys

from cloudinary.api import delete_resources_by_tag, resources_by_tag
from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url


def setup_imageup():
    cloudinary.config(
      cloud_name = 'dsvvzsmvk',  
      api_key = '821254938148261',  
      api_secret = 'irMNAC-QjZN5qIRUukTUisshyuI'  
    )

# # config
# os.chdir(os.path.join(os.path.dirname(sys.argv[0]), '.'))
# if os.path.exists('settings.py'):
#     exec(open('settings.py').read())

#DEFAULT_TAG = "python_sample_basic"

def dump_response(response):
    print("Upload response:")
    for key in sorted(response.keys()):
        print("  %s: %s" % (key, response[key]))


def upload_files(file, tag):
    print("--- Upload a local file")
    response = upload(file, tags=tag)
    dump_response(response)
    url, options = cloudinary_url(
        response['public_id'],
        format=response['format'],
        width=200,
        height=150,
        crop="fill"
    )
    print("Fill 200x150 url: " + url)
    print("")
    
def cleanup():
    response = resources_by_tag(DEFAULT_TAG)
    resources = response.get('resources', [])
    if not resources:
        print("No images found")
        return
    print("Deleting {0:d} images...".format(len(resources)))
    delete_resources_by_tag(DEFAULT_TAG)
    print("Done!")
    
if __name__ == "__main__": 
    upload_files()