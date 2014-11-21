package edu.umn.cs.recsys.uu;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;

import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.annotation.Nonnull;
import javax.inject.Inject;

import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;

import com.google.common.collect.Maps;

/**
 * User-user item scorer.
 * 
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleUserUserItemScorer extends AbstractItemScorer {
	private final UserEventDAO userDao;
	private final ItemEventDAO itemDao;

	@Inject
	public SimpleUserUserItemScorer(UserEventDAO udao, ItemEventDAO idao) {
		userDao = udao;
		itemDao = idao;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.grouplens.lenskit.ItemScorer#score(long,
	 * org.grouplens.lenskit.vectors.MutableSparseVector)
	 */
	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {

		// Nearest Neighbours count.
		int reqdNeartesNeighbours = 30;

		// Get the user's ratings
		SparseVector userVector = getUserRatingVector(user);

		// meanCentering the userVector, for finding the cosine similarity.
		SparseVector userMeanCenteredVector = meanCenterTheVector(userVector);

		// For finding the cosine similarity between users, we use the in-built
		// class of lenskit
		CosineVectorSimilarity cosineVectorSimilarity = new CosineVectorSimilarity();

		// Get the Set of items for which the score has to be computed
		LongSortedSet itemsToBeScored = scores.keyDomain();

		// Neighbourhood Map with userId and Cosine Value. A Map is used for
		// easy sorting and management and later converted into vector.
		Map<Long, Double> userSimilarityMap = Maps.newTreeMap();
		MutableSparseVector userSimilarityVector = null;

		// For all the items for this user, maintain a <itemId,
		// userSimilarityVector> structure map, later used for finding the
		// predictions.
		Map<Long, MutableSparseVector> itemVectors = Maps.newHashMap();

		// For all the items that need to be scored.
		for (Long item : itemsToBeScored) {

			// For Next item, clearing the contents of the map
			userSimilarityMap.clear();

			// Get the list of users who have rated this item.
			LongSet usersRatingTheItem = itemDao.getUsersForItem(item);

			// We find the cosines for meanCentered rating vectors and add them
			// into a map which is then sorted.
			for (Long neighbour : usersRatingTheItem) {

				// Ratings by the neighbour and meanCentering it.
				SparseVector neighbourUserVector = getUserRatingVector(neighbour);
				neighbourUserVector = meanCenterTheVector(neighbourUserVector);

				// Adding the cosines similarities into the neighbourhood map
				userSimilarityMap.put(neighbour,
						cosineVectorSimilarity.similarity(
								userMeanCenteredVector, neighbourUserVector));
			}

			// Get the Nearest N Vectors and put them into itemVectors Map for
			// easy manipulation
			userSimilarityVector = getTopNNeighboursSparseVector(user,
					reqdNeartesNeighbours, userSimilarityMap);

			// Final itemVector containing the item-wise similarity Vectors
			itemVectors.put(item, userSimilarityVector.shrinkDomain());
		}

		// Predicted Rating -
		// P(u,i) = mean-rating-of-u +
		// (Sum-over-v(cosine-sim(u,v)*(rating-for-i-vy-v -
		// mean-rating-of-v))/(Sum-over-v(cosine-sim(u,v)))
		double predictedRating = 0;

		// This is the loop structure to iterate over items to score
		for (VectorEntry e : scores.fast(VectorEntry.State.EITHER)) {
			double numerator = 0;

			// The vector containing the nearest-n similarities
			MutableSparseVector nearestNeighboursSimilarities = itemVectors
					.get(e.getKey());

			// For all the neighbours
			for (VectorEntry neighbour : nearestNeighboursSimilarities) {
				// All the ratings by the neighbour
				SparseVector neighbourRatingVector = getUserRatingVector(neighbour
						.getKey());
				double ratingForItemByNeighbour = neighbourRatingVector.get(e
						.getKey());
				double meanRatingByNeighbour = neighbourRatingVector.mean();

				numerator += ((ratingForItemByNeighbour - meanRatingByNeighbour) * neighbour
						.getValue());
			}

			// Calculating the predicted Rating and adding it to the scores
			// SparseVector
			predictedRating = userVector.mean()
					+ (numerator / nearestNeighboursSimilarities.sum());
			scores.set(e.getKey(), predictedRating);
		}
	}

	/**
	 * Converts passed userVector into a meanCentered userVector
	 * 
	 * @param userRatingVector
	 * @return {@link MutableSparseVector} a meanCentered version on the passed
	 *         userRatingVector.
	 */
	private SparseVector meanCenterTheVector(SparseVector userRatingVector) {
		double mean = userRatingVector.mean();

		// Get the Mutable Copy of the UserRatingVector for making the changes
		// in the values.
		MutableSparseVector mutUserRatingVector = userRatingVector
				.mutableCopy();
		mutUserRatingVector.add(-mean);

		return mutUserRatingVector;
	}

	/**
	 * Returns the n nearest neighbours.
	 * 
	 * @param user
	 *            - to remove the entry for the user for whom the predictions
	 *            are being calculated for
	 * @param countOfNeighbours
	 *            - number of neighbours for predictions
	 * @param neighbourhoodMap
	 *            - the map containing the <userid, cosineSimilarity> entries.
	 * 
	 * @return {@link MutableSparseVector} - the n-nearest neighbours sparse
	 *         vectors.
	 */
	private MutableSparseVector getTopNNeighboursSparseVector(long user,
			int countOfNeighbours, Map<Long, Double> neighbourhoodMap) {

		// removing the entry for the user himself, from the map.
		neighbourhoodMap.remove(user);

		// Sorting it by Values - Values being Cosine Similarities.
		neighbourhoodMap = sortByValues(neighbourhoodMap);

		// Specifying the domain of the
		MutableSparseVector userVectorCosines = MutableSparseVector
				.create(neighbourhoodMap.keySet());

		// For nearest 30 users
		Iterator<Entry<Long, Double>> iterator = neighbourhoodMap.entrySet()
				.iterator();

		// For keeping the count of top-n neighbours
		int n = 0;
		while (n < countOfNeighbours && iterator.hasNext()) {
			Entry<Long, Double> entry = iterator.next();
			userVectorCosines.set(entry.getKey(), entry.getValue());
			n++;
		}

		return userVectorCosines;
	}

	/*
	 * Sorts the passed map by value while maintaining the key.
	 * 
	 * Java method to sort Map in Java by value e.g. HashMap or Hashtable throw
	 * NullPointerException if Map contains null values It also sort values even
	 * if they are duplicates
	 */
	@SuppressWarnings("rawtypes")
	public static <K extends Comparable, V extends Comparable> Map<K, V> sortByValues(
			Map<K, V> map) {
		List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(
				map.entrySet());

		Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {
			@SuppressWarnings("unchecked")
			@Override
			public int compare(Entry<K, V> o1, Entry<K, V> o2) {
				return o2.getValue().compareTo(o1.getValue());
			}
		});

		// LinkedHashMap will keep the keys in the order they are inserted
		// which is currently sorted on natural ordering
		Map<K, V> sortedMap = new LinkedHashMap<K, V>();

		for (Map.Entry<K, V> entry : entries) {
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}

	/**
	 * Get a user's rating vector.
	 * 
	 * @param user
	 *            The user ID.
	 * @return The rating vector.
	 */
	private SparseVector getUserRatingVector(long user) {
		UserHistory<Rating> history = userDao.getEventsForUser(user,
				Rating.class);
		if (history == null) {
			history = History.forUser(user);
		}
		return RatingVectorUserHistorySummarizer.makeRatingVector(history);
	}
}