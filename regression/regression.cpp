#include <vector>
#include "../mnist.h"

float sigmoid(float x)
{
	1.0 / (1.0 + exp(-x));
}

class LinearRegression
{
public:
	void initialize();
public:
	void forward(const float* features, const float* labels, uint32_t batch)
	{
	}
	void forward(const float* feature, const float* label)
	{
		for (uint32_t i = 0; i < m_labelDimension; ++i)
		{
			float* weights = &m_weights[i * m_featureDimension];
			float z = m_biases[i];
			for (uint32_t j = 0; j < m_featureDimension; ++j)
			{
				z += feature[j] * weights[j];
			}
			m_yHats[i] = z;// sigmoid(z);
		}
		float loss = 0;
		for (uint32_t i = 0; i < m_labelDimension; ++i)
		{
			float dis = m_yHats[i] - label[i];
			loss += dis * dis;
		}

	}
private:
	std::vector<float> m_weights;
	std::vector<float> m_biases;
	std::vector<float> m_yHats;
	std::vector<float> m_weightDerivates;
	std::vector<float> m_biasDerivates;

	uint32_t m_featureDimension;
	uint32_t m_labelDimension;
};

int main()
{
	std::string path = CMAKE_SOURCE_DIR;
}
