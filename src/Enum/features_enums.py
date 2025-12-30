from enum import Enum


class FeatureEnum(str, Enum):
    ID = "id"
    VENDOR_ID = "vendor_id"
    PICKUP_DATETIME = "pickup_datetime"
    PICKUP_LONGITUDE = "pickup_longitude"
    PICKUP_LATITUDE = "pickup_latitude"
    PASSENGER_COUNT = "passenger_count"
    DROPOFF_LONGITUDE = "dropoff_longitude"
    DROPOFF_LATITUDE = "dropoff_latitude"
    STORE_AND_FWD_FLAG = "store_and_fwd_flag"
    HAVERSINE_DISTANCE = "haversine_distance"

    TRIP_DURATION = "trip_duration"
    LOG_TRIP_DURATION = "log_trip_duration"
    HOUR = "hour"
    YEAR = "year"
    DAY_OF_WEEK = "day_of_week"
    MONTH = "month"
    SEASON = "season"

    BEST_FEATURES = ['haversine_distnace', 'dropoff_longitude', 'pickup_longitude', 'dropoff_latitude', 'pickup_latitude']


