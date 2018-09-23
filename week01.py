import random
import numpy as np

class Movie(object):
    def __init__(self, title="", year=0, runtime=0):
        # The Title of the movie
        self.title = title
        # The year the movie was released
        self.year = year
        # The runtime of the movie in minutes
        self.runtime = runtime if runtime >= 0 else 0

    def __repr__(self):
        return self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins."

    def getRuntime():
        hours = self.runtime // 60
        minutes = self.runtime % 60

        return (hours, minutes)

def main():
    # Create a movie
    myMovie = Movie("Bill and Ted's Excellent Adventure", 1989, 120)

    # Print the Movie Info
    print myMovie

    # Display the list of 5 movies
    movies = create_movie_list()

    print "My 5 Movies :)"
    for movie in movies:
        print movie

    long_movies = [movie for movie in movies if movie.runtime > 150]

    print "My long movies :D"
    for movie in long_movies:
        print movie

    # Make a movie dictionary
    movieDict = {movie.title: random.randrange(0, 50, 1) / 10.0 for movie in movies}

    for title, rating in movieDict.items():
        print title + " - " + str("{:.2f}".format(rating)) + " stars!"

    movieData = get_movie_data()

    # Display number of rows
    print "Rows: " + str(len(movieData))

    # Display number of columns
    print "Columns: " + str(len(movieData[0]))

    # Print the first 2 rows (movies)
    print "First 2 Rows"
    print movieData[:2]

    # Print the last 2 columns
    for i in range(len(movieData)):
        print movieData[i][1:]

    secondColumn = []

    for i in range(len(movieData)):
        secondColumn.append(movieData[i][1])

    print secondColumn

def create_movie_list():
    # Make 5 movies and return them as a list
    return [
        Movie("Spider-Man 2", 2002, 126),
        Movie("Spider-Man 3", 2004, 138),
        Movie("The Dark Knight", 2009, 170),
        Movie("The Princess Bride", 1991, 130),
        Movie("Shrek", 2002, 120)
    ]

def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

if __name__ == "__main__":
    main()
