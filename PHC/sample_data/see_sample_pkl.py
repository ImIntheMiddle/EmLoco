"""Check the contents of a sample data file."""

import joblib

# Load the data
data = joblib.load('amass_isaac_standing_upright_slim.pkl')
data = joblib.load('amass_copycat_occlusion_v3.pkl')
# data = pickle.load(open('amass_isaac_standing_upright_slim.pkl', 'rb'))

# Print the keys
print(data.keys())
import pdb; pdb.set_trace()