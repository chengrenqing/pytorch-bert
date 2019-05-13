import os
import sys

from hashlib import sha256
import boto3
import requests
from io import open

try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse

try:
	from pathlib import Path
	PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError,ImportError):
	PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',os.path.join(os.path.expanduser("~"),'.pytorch_pretrained_bert'))

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

def url_to_filename(url,etag = None):
	url_bytes = url.encode('utf-8')
	url_hash = sha256(url_bytes)
	filename = url_hash.hexdigest()

	if etag:
		etag_bytes = etag.encode('utf-8')
		etag_hash = sha256(etag_bytes)
		filename += '.'+ etag_hash.hexdigest()

	return filename

def cached_path(url_or_filename,cache_dir=None):
	if cache_dir is None:
		cache_dir = PYTORCH_PRETRAINED_BERT_CACHE

	if sys.version_info[0] == 3 and isinstance(url_or_filename,Path):
		url_or_filename = str(url_or_filename)

	if sys.version_info[0] == 3 and isinstance(cache_dir,Path):
		cache_dir = str(cache_dir)

	parsed = urlparse(url_or_filename)

	if parsed.scheme in ('http','https','s3'):
		return get_from_cache(url_or_filename,cache_dir)
	elif os.path.exists(url_or_filename):
		return url_or_filename
	elif parsed.scheme == '':
		raise EnvironmentError("File {} not found".format(url_or_filename))
	else:
		raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

def s3_etag(url):
	s3_resource = boto3.resource("s3")
	bucket_name,s3_path = split_s3_path(url)
	s3_object = s3_resource.Object(bucket_name,s3_path)
	return s3_object.e_tag

def get_from_cache(url,cache_dir=None):
	if cache_dir is None:
		cache_dir = PYTORCH_PRETRAINED_BERT_CACHE

	if sys.version_info[0] == 3 and isinstance(cache_dir,Path):
		cache_dir = str(cache_dir)

	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	if url.startswith("s3://"):
		etag = s3_etag(url) #call
	else:
		try:
			response = requests.head(url,allow_redirects=True)
			if response.status_code != 200:
				etag = None
			else:
				etag = response.headers.get("ETag")
		except EnvironmentError:
			etag = None

	if sys.version_info[0] == 2 and etag is not None:
		etag = etag.decode('utf-8')

	filename = url_to_filename(url,etag) #call

	cache_path = os.path.join(cache_dir,filename)

	if not os.path.exists(cache_path) and etag is None:
		matching_files = fnmatch.filter(os.listdir(cache_dir),filename + '.*')
		matching_files = list(filter(lambda s: not s.endswith('.json'),matching_files))
		if matching_files:
			cache_path = os.path.join(cache_dir,matching_files[-1])

	if not os.path.exists(cache_path):
		with tempfile.NamedTemporaryFile() as temp_file:
			logger.info("%s not found in cache,downloading to %s",url,temp_file.name)

			if url.startswith("s3://"):
				s3_get(url,temp_file)
			else:
				http_get(url,temp_file)

			temp_file.flush()
			temp_file.seek(0)

			logger.info("creating metadata file for %s",cache_path)
			meta = {'url':url,'etag':etag}
			meta_path = cache_path + '.json'
			with open(meta_path,'w') as meta_file:
				output_string = json.dumps(meta)
				if sys.version_info[0] == 2 and isinstance(output_string,str):
					output_string = unicode(output_string,'utf-8')
				meta_file.write(output_string)

			logger.info("removing temp file %s",temp_file.name)

	return cache_path


