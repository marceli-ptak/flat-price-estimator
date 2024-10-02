import os
import pickle
import numpy as np
import psycopg2
from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from memory_profiler import profile
import gzip
import subprocess
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')
port = os.getenv('DB_PORT')

# Ensure port is not None
if port is None:
    raise ValueError("DB_PORT environment variable is not set")

# Check if the database exists, if not create it
conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{database}'")
exists = cur.fetchone()
if not exists:
    cur.execute(f"CREATE DATABASE {database}")
cur.close()
conn.close()

backup_file = 'location_db.sql'
# Command to restore the database
command = f'pg_restore --if-exists --dbname=postgresql://{user}:{password}@{host}:{port}/{database} -c -v {backup_file}'
# Execute the command
subprocess.run(command, shell=True, check=True)

# Create the SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
# Create a configured "Session" class
Session = sessionmaker(bind=engine)
# Create a session
session = Session()

# Loading the predictive model
with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

# Main page with the form
@app.route('/')
@profile
def index():
    return render_template('index.html')

@app.route('/get_cities')
@profile
def get_cities():
    """
    Fetches the list of cities from the database.

    Returns:
        Response: A JSON response containing a list of cities,
                  where each city is represented as a dictionary
                  with 'id' and 'name' keys.
    """
    cities = session.execute(text("SELECT id, name FROM cities")).fetchall()
    city_list = [{'id': city.id, 'name': city.name} for city in cities]
    return jsonify(city_list)

@app.route('/get_districts/<city_id>')
@profile
def get_districts(city_id):
    """
        Fetches the list of districts for a given city.

        Args:
            city_id (int): The ID of the city for which to fetch districts.

        Returns:
            Response: A JSON response containing a list of districts,
                      where each district is represented as a dictionary
                      with 'id' and 'name' keys.
    """
    districts = session.execute(text("SELECT id, name FROM districts WHERE city_id = :city_id"),{'city_id': city_id}).fetchall()
    district_list = [{'id': district.id, 'name': district.name} for district in districts]
    return jsonify(district_list)

@app.route('/get_neighborhoods/<district_id>')
@profile
def get_neighborhoods(district_id):
    """
    Fetches the list of neighborhoods for a given district.

    Args:
        district_id (int): The ID of the district for which to fetch neighborhoods.

    Returns:
        Response: A JSON response containing a list of neighborhoods,
                  where each neighborhood is represented as a dictionary
                  with 'id' and 'name' keys.
    """
    neighborhoods = session.execute(text("SELECT id, name FROM neighborhoods WHERE district_id = :district_id"),{'district_id': district_id}).fetchall()
    neighborhood_list = [{'id': neighborhood.id, 'name': neighborhood.name} for neighborhood in neighborhoods]
    return jsonify(neighborhood_list)

# Endpoint for price prediction
@app.route('/predict', methods=['POST'])
@profile
def predict():
    """
     Handles the prediction of apartment prices based on user input from a form.

     This function receives data from a POST request, processes the input,
     and uses a pre-trained model to predict the price of an apartment.
     It also fetches average prices for the selected city, district, and neighborhood
     from the database and renders the result on a web page.

     Returns:
         Response: A rendered HTML template with the predicted price and average prices
                   for the selected city, district, and neighborhood.
     """
    if request.method == 'POST':
        # Receiving data from the form
        city_id = request.form.get('encoded_city')
        district_id = request.form.get('encoded_district')
        neighborhood_id = request.form.get('encoded_neighborhood')
        area = float(request.form.get('area'))
        floor = int(request.form.get('floor'))
        building_height = int(request.form.get('building_height'))
        lift = 1 if request.form.get('lift') == 'yes' else 0
        number_of_rooms = int(request.form.get('number_of_rooms'))
        primary_market = 1 if request.form.get('primary_market') == 'primary' else 0
        ready_to_live = 1 if request.form.get('ready_to_live') == 'ready for occupancy' else 0
        amenities = request.form.getlist('amenities')
        ac = 1 if 'AC' in amenities else 0
        build_year = int(request.form.get('build_year'))  # Typ int
        in_1946_2000 = 1 if 1946 <= build_year <= 2000 else 0
        in_2001_2030 = 1 if 2001 <= build_year <= 2030 else 0

        if not city_id or not district_id or not neighborhood_id:
            return "Error: City, District, and Neighborhood must be selected", 400

        # Processing logic for the amenities field
        if len(amenities) <= 1:
            amenities_value = 0
        elif 2 <= len(amenities) <= 3:
            amenities_value = 1
        else:
            amenities_value = 2

        # Logarithmic transformation of the 'area' variable (log1p, i.e., log(1 + area))
        log_area = np.log1p(area)

        # Preparing data for the model (in order of columns in the model)
        input_data = [[
            neighborhood_id,  # encoded_neighborhood
            log_area,  # area after log transformation
            district_id,  # encoded_district
            in_2001_2030,  # building age between (2001,2030)
            city_id,  # encoded_city
            building_height,  # building height
            floor,  # floor number
            ready_to_live,  # condition
            primary_market,  # type of market
            lift,  # lift
            number_of_rooms,  # number of rooms
            ac, # air condition
            in_1946_2000, # building age between (1946,2000)
            amenities_value  # amenities
        ]]

        # Predicting the price based on the model
        predicted_price_log = model.predict(input_data)[0]

        # Appropriate inverse of the logarithmic transformation
        predicted_price = round(np.exp(predicted_price_log))

        # Fetching average prices from the database
        city_avg_price = session.execute(text("SELECT city_avg_price FROM cities WHERE id = :city_id"),
                                         {'city_id': city_id}).scalar()
        city_name = session.execute(text("SELECT name FROM cities WHERE id = :city_id"),
                                         {'city_id': city_id}).scalar()
        district_avg_price = session.execute(text("SELECT district_avg_price FROM districts WHERE id = :district_id"),
                                             {'district_id': district_id}).scalar()
        district_name = session.execute(text("SELECT name FROM districts WHERE id = :district_id"),
                                             {'district_id': district_id}).scalar()
        neighborhood_avg_price = session.execute(text("SELECT neighborhood_avg_price FROM neighborhoods WHERE id = :neighborhood_id"),
                                            {'neighborhood_id': neighborhood_id}).scalar()
        neighborhood_name = session.execute(text("SELECT name FROM neighborhoods WHERE id = :neighborhood_id"),
                                            {'neighborhood_id': neighborhood_id}).scalar()

        return render_template('result.html', price=predicted_price, city_avg_price=city_avg_price,
                               district_avg_price=district_avg_price, neighborhood_avg_price=neighborhood_avg_price,
                               city_name=city_name, district_name=district_name, neighborhood_name=neighborhood_name)

@app.teardown_appcontext
def shutdown_session(exception=None):
    """
        Closes the SQLAlchemy session.

        This function is called automatically when the Flask application context is torn down.
        It ensures that the SQLAlchemy session is properly closed to release database connections
        and other resources.

        Args:
            exception (Exception, optional): An optional exception that may have occurred during
                                             the request handling. Defaults to None.
    """
    session.close()

if __name__ == '__main__':
    app.run(debug=True)