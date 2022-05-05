import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bs4 as bs
import urllib.request
import pickle

main_df = pd.read_csv('main_df.csv')
cv = CountVectorizer()
vector = cv.fit_transform(main_df['comb']).toarray()
similarity_ = cosine_similarity(vector)

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)


def movie_recommender(movie):
    movie = str(movie).lower().strip()
    if movie == "bāhubali: the beginning":
        movie = "baahubali: the beginning"
    if movie == "like stars on earth":
        movie = "taare zameen par"
    l = main_df['movie_title'].unique()
    if movie not in l:
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    index = main_df[main_df['movie_title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity_[index])), reverse=True, key=lambda x: x[1])
    rec_movies = [main_df.iloc[i[0]].movie_title for i in distances[1:11]]
    return rec_movies


def genre_recommender(genre):
    genre_vector = cv.fit_transform(main_df['genres']).toarray()
    genre_similarity = cosine_similarity(genre_vector)
    sorted(list(enumerate(genre_similarity[0])),
           reverse=True, key=lambda x: x[1])[1:6]
    index = main_df[main_df['genres'] == genre].index[0]
    distances = sorted(
        list(enumerate(genre_similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:11]:
        print(main_df.iloc[i[0]].movie_title)


def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

# https://api.themoviedb.org/3/search/movie?api_key=a6270c31daee789ef6c54ceedce0a34c&query=taare+zameen+par


def get_suggestions():
    return list(main_df['movie_title'].str.capitalize())


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('index.html', suggestions=suggestions)


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = movie_recommender(movie)
    return rc if type(rc) == str else "---".join(rc)


# @app.route("/similar_genre", methods=["POST"])
# def similar_genre():
#     genre = request.form['name']
#     rc = genre_recommender(genre)
#     return rc if type(rc) == str else "---".join(rc)

# @app.route("/genre_recommend", methods=['POST'])
# def genre_recommend():
#     rec_movies = request.form['rec_movies']
#     rec_posters = request.form['rec_posters']
#     return render_template("genre_recommend.html", rec_movies=rec_movies, rec_posters=rec_posters)


@app.route("/movie_recommend", methods=["POST"])
def movie_recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i]
                   for i in range(len(rec_posters))}

    casts = {cast_names[i]: [cast_ids[i], cast_chars[i],
                             cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i],
                                    cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen(
        f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt').read()

    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []  # list of reviews
    reviews_status = []  # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i]
                     for i in range(len(reviews_list))}

    # passing all the data to the html file
    return render_template('movie_recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
                           vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
                           movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)


if __name__ == '__main__':
    app.run(debug=True)
