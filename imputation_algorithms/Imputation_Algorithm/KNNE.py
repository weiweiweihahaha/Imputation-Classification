from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import collections as cl
import pandas as pd
import numpy as np
import random

class ComponentClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, criterion='entropy', attributeFilteringCoeff=None, bootstrap=True, s=1.0, minkowskyMin=2, minkowskyMax=2, 
			kMin=5, kMax=5, votingType='majority_componentClf_majority', random_state=None):
		self.criterion = criterion
		self.attributeFilteringCoeff = attributeFilteringCoeff
		self.bootstrap = bootstrap
		self.s = s
		self.minkowskyMin = minkowskyMin
		self.minkowskyMax = minkowskyMax
		self.kMin = kMin
		self.kMax = kMax
		self.votingType = votingType
		self.random_state = random_state

	def _cleanFeatures(self, X):

		# Clean data to include only features from feature subset
		droppedColumns = [index for index in range(len(self.useFeatures_)) if self.useFeatures_[index] == 0]
		X = pd.DataFrame(X)
		X = X.drop(droppedColumns, axis=1)

		return X

	def fit(self, X, y):

		# Check that parameters are formatted properly
		X, y = check_X_y(X, y)
		assert ((self.attributeFilteringCoeff == None) or type(self.attributeFilteringCoeff) == float), "attributeFilteringCoeff parameter must be None or float"
		assert (type(self.bootstrap) == bool), "bootstrap parameter must be bool"
		assert ((self.s == None) or type(self.s) == float), "s parameter must be None or float"
		assert (type(self.minkowskyMin) == int), "minkowskyMin parameter must be int"
		assert (type(self.minkowskyMax) == int), "minkowskyMax parameter must be int"
		assert (type(self.kMin) == int), "kMin parameter must be int"
		assert (type(self.kMax) == int), "kMax parameter must be int"
		assert (type(self.votingType) == str), "votingType parameter must be str"
		assert ((self.random_state == None) or (type(self.random_state) == int)), "random_state parameter must be None or int"
		assert (self.criterion in ['entropy', 'gini']), "criterion parameter must be \'entropy\' or \'gini\'"		
		assert ((self.attributeFilteringCoeff == None) or (self.attributeFilteringCoeff > 0)), "attributeFilteringCoeff parameter must be greater than or equal to 0"
		assert ((self.s == None) or ((0 < self.s) and (self.s <= 1))), "s parameter must be greater than 0 and less than or equal to 1"
		assert (0 < self.minkowskyMin), "minkowskyMin parameter must be greater than 0"
		assert (0 < self.minkowskyMax), "minkowskyMax parameter must be greater than 0"
		assert (self.minkowskyMin <= self.minkowskyMax), "minkowskyMin parameter must be less than or equal to minkowskyMax parameter"
		assert (1 <= self.kMin), "kMin parameter must be greater than or equal to 1"
		assert (1 <= self.kMax), "kMax parameter must be greater than or equal to 1"
		assert (self.kMin <= self.kMax), "kMin parameter must be less than or equal to kMax parameter"
		assert (self.votingType in ['majority_componentClf_majority', 'majority_componentClf_bordaOrder', 'majority_componentClf_bordaDistance', 'majority_allK', 
				'bordaCount_allK_bordaOrder', 'bordaCount_allK_bordaDistance']), "votingType parameter must be of a recognized type"

		# Seed random number generator
		random.seed(self.random_state)

		# Apply attribute filtering
		if self.attributeFilteringCoeff != None:

			# Get information gain
			igClf = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)
			igClf.fit(X, y)
			featureInformationGains = igClf.feature_importances_

			# Filter which features according to information gain
			avgInformationGains = np.mean(featureInformationGains)
			f = self.attributeFilteringCoeff * avgInformationGains
			validFeatures = [0 if gain < f else 1 for gain in featureInformationGains]

			# If no features made the cutoff, set a random feature as valid
			if 1 not in validFeatures:
				randomIndex = random.choice([i for i in range(len(validFeatures))])
				validFeatures[randomIndex] = 1
		else:
			validFeatures = [1 for i in range(len(X[0]))]

		# Apply feature selection
		validIndicies = [index for index in range(len(validFeatures)) if validFeatures[index] == 1]
		floatNumChoose = self.s * len(validIndicies)
		decimalNumChoose = str(floatNumChoose).split('.')[1]
		potentialNumChoose = random.choice([int(floatNumChoose), int(floatNumChoose + 1)]) if decimalNumChoose == '5' else round(floatNumChoose)
		numChoose = 1 if potentialNumChoose == 0 else potentialNumChoose
		chosenIndicies = random.sample(validIndicies, numChoose)
		useFeatures = [1 if index in chosenIndicies else 0 for index in range(len(validFeatures))]
		self.useFeatures_ = useFeatures

		# Get a bootstrapped sample of the data
		if self.bootstrap == True:
			X, y = resample(X, y, random_state=self.random_state)

		# Clean data to include only features from feature subset
		X = self._cleanFeatures(X)
		n_samples = len(y)
		if n_samples < 1:
			raise ValueError("ComponentClassifier requires at least one training sample")

		# Generate an odd K for even number class problem and viceversa, but never exceed train size.
		max_k = min(self.kMax, n_samples)
		min_k = min(self.kMin, max_k)
		candidates = [k for k in range(min_k, max_k + 1)]
		if not candidates:
			candidates = [1]
		if len(unique_labels(y)) % 2 == 0:
			parity_candidates = [k for k in candidates if k % 2 == 1]
		else:
			parity_candidates = [k for k in candidates if k % 2 == 0]
		usable_candidates = parity_candidates if parity_candidates else candidates
		self.n_neighbors_ = usable_candidates[0] if len(usable_candidates) == 1 else random.choice(usable_candidates)

		# Generate a value for the order of the Minkowsky distance
		minkowskyOrder = random.randint(self.minkowskyMin, self.minkowskyMax)

		# Initialize and train KNN classifier on feature subset
		knn = KNeighborsClassifier(n_neighbors=self.n_neighbors_, p=minkowskyOrder)
		knn.fit(X, y)
		self.y_ = y
		self.knn_ = knn

		# Create class variable of classification class values
		self.classes_ = unique_labels(y)

		return self

	def predict(self, X):

		# Check that classifier has been fitted
		try:
			getattr(self, "knn_")
		except AttributeError:
			raise NotFittedError("You must fit classifer before using predict!")

		# Convert X to numpy array
		X = check_array(X)

		# Clean data to include only features from feature subset
		X = self._cleanFeatures(X)

		# Generate predictions based on voting method
		if self.votingType == 'majority_componentClf_majority':
			predictions = self.knn_.predict(X)
			return predictions

		elif self.votingType == 'majority_componentClf_bordaOrder':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []
			for indexList in indicies:
				counter = cl.Counter()
				for i in range(len(indexList)):
					index = indexList[i]
					iclass = self.y_[index]
					points = len(indexList) - i
					counter[iclass] += points
				choice = counter.most_common(1)[0][0]
				predictions.append(choice)
			return predictions

		elif self.votingType == 'majority_componentClf_bordaDistance':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []
			for i in range(len(indicies)):
				indexList = indicies[i]
				distanceList = distances[i]
				counter = cl.Counter()
				for j in range(len(indexList)):
					index = indexList[j]
					iclass = self.y_[index]
					distance = distanceList[j]
					counter[iclass] += (1.0 /  (distance + 1.0))
				choice = counter.most_common(1)[0][0]
				predictions.append(choice)
			return predictions

		elif self.votingType == 'majority_allK':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = [[self.y_[index] for index in indexList] for indexList in indicies]
			return predictions

		elif self.votingType == 'bordaCount_allK_bordaOrder':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []
			for indexList in indicies:
				counter = cl.Counter()
				for i in range(len(indexList)):
					index = indexList[i]
					iclass = self.y_[index]
					points = len(indexList) - i
					counter[iclass] += points
				predictions.append(counter)
			return predictions

		elif self.votingType == 'bordaCount_allK_bordaDistance':
			(distances, indicies) = self.knn_.kneighbors(X, self.n_neighbors_)
			predictions = []

			for i in range(len(indicies)):
				indexList = indicies[i]
				distanceList = distances[i]
				counter = cl.Counter()
				for j in range(len(indexList)):
					index = indexList[j]
					iclass = self.y_[index]
					distance = distanceList[j]
					counter[iclass] += (1.0 / (distance + 1.0))
				predictions.append(counter)
			return predictions

	def score(self, X, y, sample_weight=None):

		# Check that classifier has been fitted
		try:
			getattr(self, "knn_")
		except AttributeError:
			raise NotFittedError("You must fit classifer before using predict!")

		# Convert X to numpy array
		X = check_array(X)

		# Clean data to include only features from feature subset
		X = self._cleanFeatures(X)

		return self.knn_.score(X, y, sample_weight=sample_weight)


def _fallback_fill(df):
	result = df.copy()
	for column in result.columns:
		series = result[column]
		if not series.isnull().any():
			continue
		numeric = pd.to_numeric(series, errors='coerce')
		if numeric.notna().any():
			fill_value = numeric.mean()
		else:
			modes = series.mode(dropna=True)
			fill_value = modes.iloc[0] if not modes.empty else 0
		result[column] = series.fillna(fill_value)
	return result.to_numpy(dtype=float)


def KNNE(df, random_state=42):

	if not df.isnull().to_numpy().any():
		return df.to_numpy(dtype=float)

	X = df.to_numpy(dtype=float)
	imputed = X.copy()

	for label_column in range(X.shape[1]):
		missing_targets = np.isnan(X[:, label_column])
		if not np.any(missing_targets):
			continue

		feature_columns = [i for i in range(X.shape[1]) if i != label_column]
		feature_block = X[:, feature_columns]
		train_mask = (~missing_targets) & (~np.isnan(feature_block).any(axis=1))
		test_mask = missing_targets

		if not np.any(train_mask):
			fallback = _fallback_fill(df)
			imputed[:, label_column] = fallback[:, label_column]
			continue

		X_train = np.nan_to_num(X[train_mask][:, feature_columns], nan=0.0)
		y_train = X[train_mask, label_column]
		X_test = np.nan_to_num(X[test_mask][:, feature_columns], nan=0.0)

		if X_test.shape[0] == 0:
			continue

		clf = ComponentClassifier(
			criterion='entropy',
			attributeFilteringCoeff=0.5,
			bootstrap=True,
			s=0.8,
			minkowskyMin=1,
			minkowskyMax=3,
			kMin=5,
			kMax=10,
			votingType='majority_componentClf_majority',
			random_state=random_state,
		)
		try:
			clf.fit(X_train, y_train.astype('str'))
			predictions = clf.predict(X_test)
			imputed[test_mask, label_column] = pd.to_numeric(pd.Series(predictions), errors='coerce').to_numpy()
		except Exception:
			fallback = _fallback_fill(df)
			imputed[test_mask, label_column] = fallback[test_mask, label_column]

	result = pd.DataFrame(imputed, columns=df.columns).fillna(df.mean(axis=0, skipna=True)).fillna(0.0)
	return result.to_numpy(dtype=float)
