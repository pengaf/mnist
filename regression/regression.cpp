#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include "../mnist.h"

float sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

template<bool softmax = false>
class LogisticRegression
{
public:
	LogisticRegression(uint32_t featureDimension, uint32_t numClassify)
	{
		m_featureDimension = featureDimension;
		m_numClassify = numClassify;
		m_weights.resize(featureDimension * numClassify);
		m_biases.resize(numClassify);
		m_yHats.resize(numClassify);
		m_weightDerivates.resize(featureDimension * numClassify);
		m_biasDerivates.resize(numClassify);
		m_sumWeightDerivates.resize(featureDimension * numClassify);
		m_sumBiasDerivates.resize(numClassify);
		for (auto& weight : m_weights)
		{
			weight = rand() / (float(RAND_MAX)) * 0.3f;
		}
		for (auto& bias : m_biases)
		{
			bias = rand() / (float(RAND_MAX));
		}
	}

public:
	void miniBatch(const uint8_t* features, const uint8_t* labels, uint32_t batchSize, float eta)
	{
		for (uint32_t i = 0; i < m_numClassify; ++i)
		{
			for (uint32_t j = 0; j < m_featureDimension; ++j)
			{
				m_sumWeightDerivates[i * m_featureDimension + j] = 0;
			}
			m_sumBiasDerivates[i] = 0;
		}
		for (uint32_t b = 0; b < batchSize; ++b)
		{
			forward(features + b * m_featureDimension, labels[b]);
			for (uint32_t i = 0; i < m_numClassify; ++i)
			{
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					m_sumWeightDerivates[i * m_featureDimension + j] += m_weightDerivates[i * m_featureDimension + j];
				}
				m_sumBiasDerivates[i] += m_biasDerivates[i];
			}
		}
		for (uint32_t i = 0; i < m_numClassify; ++i)
		{
			for (uint32_t j = 0; j < m_featureDimension; ++j)
			{
				float weightDerivate = m_sumWeightDerivates[i * m_featureDimension + j] / batchSize;
				m_weights[i * m_featureDimension + j] -= weightDerivate * eta;
			}
			float biasDerivate = m_sumBiasDerivates[i] / batchSize;
			m_biases[i] -= biasDerivate * eta;
		}
	}
	void forward(const uint8_t* feature, uint8_t label)
	{
		if (softmax)
		{
			float sumExpZ = 0;
			for (uint32_t i = 0; i < m_numClassify; ++i)
			{
				float* weights = &m_weights[i * m_featureDimension];
				float z = m_biases[i];
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					z += feature[j] / 255.0f * weights[j];
				}
				float expz = exp(z);
				m_yHats[i] = z;
				sumExpZ += expz;
			}
			for (uint32_t i = 0; i < m_numClassify; ++i)
			{
				float yHat = exp(m_yHats[i]) / sumExpZ;
				float yLabel = (i == label ? 1.0 : 0);
				float zDerivate = (yHat - yLabel);
				float* weightDerivates = &m_weightDerivates[i * m_featureDimension];
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					weightDerivates[j] = zDerivate * feature[j] / 255.0f;
				}
				m_biasDerivates[i] = zDerivate;
			}
		}
		else
		{
			for (uint32_t i = 0; i < m_numClassify; ++i)
			{
				float* weights = &m_weights[i * m_featureDimension];
				float z = m_biases[i];
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					z += feature[j] / 255.0f * weights[j];
				}
				m_yHats[i] = sigmoid(z);

				float yLabel = (i == label ? 1.0 : 0);
				float zDerivate = m_yHats[i] - yLabel;
				float* weightDerivates = &m_weightDerivates[i * m_featureDimension];
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					weightDerivates[j] = zDerivate * feature[j] / 255.0f;
				}
				m_biasDerivates[i] = zDerivate;
			}
		}
	}
	uint8_t evaluate(const uint8_t* feature)
	{
		if (softmax)
		{
			for (uint32_t i = 0; i < m_numClassify; ++i)
			{
				float* weights = &m_weights[i * m_featureDimension];
				float z = m_biases[i];
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					z += feature[j] / 255.0f * weights[j];
				}
				//float expz = exp(z);
				m_yHats[i] = z;
			}
		}
		else
		{
			for (uint32_t i = 0; i < m_numClassify; ++i)
			{
				float* weights = &m_weights[i * m_featureDimension];
				float z = m_biases[i];
				for (uint32_t j = 0; j < m_featureDimension; ++j)
				{
					z += feature[j] / 255.0f * weights[j];
				}
				m_yHats[i] = sigmoid(z);
			}
		}
		uint32_t index = std::distance(m_yHats.begin(), std::max_element(m_yHats.begin(), m_yHats.end()));
		return index;
	}

	float test(const uint8_t* features, const uint8_t* labels, uint32_t count)
	{
		uint32_t errorCount = 0;
		for (uint32_t i = 0; i < count; ++i)
		{
			uint8_t yHat = evaluate(features + i * m_featureDimension);
			uint8_t yLabel = *(labels + i);
			if (yHat != yLabel)
			{
				++errorCount;
			}
		}
		return float(errorCount) / float(count);
	}
public:
	uint32_t m_featureDimension;
	uint32_t m_numClassify;
	std::vector<float> m_weights;
	std::vector<float> m_biases;
	std::vector<float> m_yHats;
	std::vector<float> m_weightDerivates;
	std::vector<float> m_biasDerivates;
	std::vector<float> m_sumWeightDerivates;
	std::vector<float> m_sumBiasDerivates;
};

int main()
{
	std::string path = CMAKE_SOURCE_DIR;

	MnistImageHeader trainImageHeader;
	MnistLabelHeader trainLabelHeader;
	MnistImageHeader testImageHeader;
	MnistLabelHeader testLabelHeader;
	std::vector<uint8_t> trainImages;
	std::vector<uint8_t> trainLabels;
	std::vector<uint8_t> testImages;
	std::vector<uint8_t> testLabels;
	bool b1 = ReadImageData(trainImageHeader, trainImages, path + "/data/train-images.idx3-ubyte");
	bool b2 = ReadLabelData(trainLabelHeader, trainLabels, path + "/data/train-labels.idx1-ubyte");
	bool b3 = ReadImageData(testImageHeader, testImages, path + "/data/t10k-images.idx3-ubyte");
	bool b4 = ReadLabelData(testLabelHeader, testLabels, path + "/data/t10k-labels.idx1-ubyte");
	if (!(b1 && b2 && b3 && b4))
	{
		return 0;
	}

	uint32_t featureDimension = trainImageHeader.columnCount * trainImageHeader.rowCount;
	uint32_t validationCount = trainImageHeader.imageCount / 10;
	uint32_t trainCount = trainImageHeader.imageCount - validationCount;
	uint32_t testCount = testImageHeader.imageCount;

	LogisticRegression<true> logisticRegression(featureDimension, 10);

	uint32_t batchSize = 10;
	uint32_t numBatch = trainCount / batchSize;
	float eta = 0.003;
	uint32_t epoch = 40;


	uint32_t errorCount = 0;
	for (uint32_t i = 0; i < validationCount; ++i)
	{
		uint8_t yHat = logisticRegression.evaluate(trainImages.data() + (trainCount + i) * featureDimension);
		uint8_t yLabel = *(trainLabels.data() + trainCount + i);
		if (yHat != yLabel)
		{
			++errorCount;
		}
	}

	printf("init error %f, %f, %f\n", 
		logisticRegression.test(trainImages.data(), trainLabels.data(), trainCount) * 100,
		logisticRegression.test(trainImages.data() + trainCount * featureDimension, trainLabels.data() + trainCount, validationCount) * 100,
		logisticRegression.test(testImages.data(), testLabels.data(), testCount) * 100);


	for (uint32_t e = 0; e < epoch; ++e)
	{
		for (uint32_t b = 0; b < numBatch; ++b)
		{
			logisticRegression.miniBatch(trainImages.data() + b * batchSize * featureDimension, trainLabels.data() + b * batchSize, batchSize, eta);
		}
		printf("%d: error %f, %f, %f\n", e + 1,
			logisticRegression.test(trainImages.data(), trainLabels.data(), trainCount) * 100,
			logisticRegression.test(trainImages.data() + trainCount * featureDimension, trainLabels.data() + trainCount, validationCount) * 100,
			logisticRegression.test(testImages.data(), testLabels.data(), testCount) * 100);
	}

	std::unordered_map<uint32_t, uint32_t> errors;

	for (uint32_t i = 0; i < testImageHeader.imageCount; ++i)
	{
		uint8_t yHat = logisticRegression.evaluate(testImages.data() + i * featureDimension);
		uint8_t yLabel = *(testLabels.data() + i);
		if (yHat != yLabel)
		{
			++errorCount;
			uint32_t key = (uint32_t(yHat) << 8) | uint32_t(yLabel);
			errors.insert(std::make_pair(key, 1)).first->second++;
		}
	}

	std::multimap<uint32_t, uint32_t> sortedErrors;
	for (auto it = errors.begin(); it != errors.end(); ++it)
	{
		sortedErrors.insert(std::make_pair(it->second, it->first));
	}
	for (auto it = sortedErrors.begin(); it != sortedErrors.end(); ++it)
	{
		uint32_t key = it->second;
		uint32_t yLabel = key & 0xff;
		uint32_t yHat = key >> 8;
		//printf("errors: %d, %d, %d\n", yLabel, yHat, it->first);
	}
	//mnist2bmp(path + "/data/t10k-images.bmp", path + "/data/t10k-images.idx3-ubyte");
}
