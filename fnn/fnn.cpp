#include <fstream>
#include <vector>

const uint32_t mnist_image_header_flag = 0x00000803;
const uint32_t mnist_label_header_flag = 0x00000801;

struct MnistImageHeader
{
	uint32_t magicNumber;
	uint32_t imageCount;
	uint32_t rowCount;
	uint32_t columnCount;
};

struct MnistLabelHeader
{
	uint32_t magicNumber;
	uint32_t labelCount;
};

uint32_t ConvertEndian(uint32_t n)
{
	return ((n << 24) & 0xFF000000) | ((n << 8) & 0x00FF0000) | ((n >> 8) & 0x0000FF00) | ((n >> 24) & 0x000000FF);
}

bool ReadImageData(MnistImageHeader& header, std::vector<char>& data, const char* fileName)
{
	std::ifstream file(fileName, std::ios::binary);
	if (!file.is_open())
	{
		return false;
	}
	uint32_t tmp;
	file.read((char*)&tmp, sizeof(uint32_t));
	header.magicNumber = ConvertEndian(tmp);
	if (header.magicNumber != 0x00000803)
	{
		return false;
	}
	file.read((char*)&tmp, sizeof(uint32_t));
	header.imageCount = ConvertEndian(tmp);
	file.read((char*)&tmp, sizeof(uint32_t));
	header.rowCount = ConvertEndian(tmp);
	file.read((char*)&tmp, sizeof(uint32_t));
	header.columnCount = ConvertEndian(tmp);

	uint32_t dataSize = header.imageCount*header.rowCount*header.columnCount;
	data.resize(dataSize);
	file.read(data.data(), dataSize);
	return true;
}

bool ReadLabelData(MnistLabelHeader& header, std::vector<char>& data, const char* fileName)
{
	std::ifstream file(fileName, std::ios::binary);
	if (!file.is_open())
	{
		return false;
	}
	uint32_t tmp;
	file.read((char*)&tmp, sizeof(uint32_t));
	header.magicNumber = ConvertEndian(tmp);
	if (header.magicNumber != 0x00000803)
	{
		return false;
	}
	file.read((char*)&tmp, sizeof(uint32_t));
	header.labelCount = ConvertEndian(tmp);

	uint32_t dataSize = header.labelCount;
	data.resize(dataSize);
	file.read(data.data(), dataSize);
	return true;
}

//class ActivationFunction
//{
//public:
//	virtual void forward(float* outputs, const float* inputs, uint32_t dim) = 0;
//	virtual void backward(float* outputs, const float* inputs, uint32_t dim) = 0;
//};
//
//class Sigmoid : public ActivationFunction
//{
//public:
//	virtual void forward(float* outputs, const float* inputs, uint32_t dim)
//	{
//		for (uint32_t i = 0; i < dim; ++i)
//		{
//			outputs[i] = 1.0 / (1.0 + exp(-inputs[i]));
//		}
//	}
//	virtual void backward(float* outputs, const float* inputs, uint32_t dim)
//	{
//	}
//public:
//	static Sigmoid* GetInstance()
//	{
//		static Sigmoid s_instance;
//		return &s_instance;
//	}
//};
//
//class Softmax : public ActivationFunction
//{
//public:
//	virtual void forward(float* outputs, const float* inputs, uint32_t dim)
//	{
//		float total = 0;
//		for (uint32_t i = 0; i < dim; ++i)
//		{
//			float tmp = exp(inputs[i]);
//			outputs[i] = tmp;
//			total += tmp;
//		}
//		float rcpTotal = 1.0 / total;
//		for (uint32_t i = 0; i < dim; ++i)
//		{
//			outputs[i] *= rcpTotal;
//		}
//	}
//	virtual void backward(float* outputs, const float* inputs, uint32_t dim)
//	{
//	}
//public:
//	static Sigmoid* GetInstance()
//	{
//		static Sigmoid s_instance;
//		return &s_instance;
//	}
//};
//

class LossFunction
{
public:
	LossFunction(uint32_t dimension)
	{
		m_dimension = dimension;
		m_losses.resize(dimension);
		m_derivates.resize(dimension);
	}
	uint32_t dimension() const
	{
		return m_dimension;
	}
	const float* derivates()
	{
		return m_derivates.data();
	}
public:
	virtual void forward(const float* yHat, const float* yLabel) = 0;
protected:
	uint32_t m_dimension;
	std::vector<float> m_losses;
	std::vector<float> m_derivates;
};

class MeanSquareError : public LossFunction
{
public:
	//float forward(float* y, float* yHat, uint32_t batch) override
	//{
	//	float totalLoss = 0;
	//	for (uint32_t i = 0; i < batch; ++i)
	//	{
	//		float dis = y - yHat;
	//		float loss = dis * dis;
	//		totalLoss += loss;
	//	}
	//	float c = totalLoss / (2 * batch);
	//	return c;
	//}
	virtual float derivate(float yHat, float yLabel) override
	{
		float res = 2.0*(yHat - yLabel);
		return res;
	}
//public:
//	static MeanSquareError* GetInstance()
//	{
//		static MeanSquareError s_instance;
//		return &s_instance;
//	}
};

class Layer
{
public:
	Layer(uint32_t numInputs, uint32_t numOutputs) :
		m_numInputs(numInputs),
		m_numOutputs(numOutputs)
	{
		m_features.resize(numOutputs);
	}
public:
	virtual void forward(const float* inputs) = 0;
	virtual void backward(const float* inputs) = 0;
public:
	uint32_t numInputs() const
	{
		return m_numInputs;
	}
	uint32_t numOutputs() const
	{
		return m_numOutputs;
	}
	const float* outputFeatures() const
	{
		return m_features.data();
	}
protected:
	uint32_t m_numInputs;
	uint32_t m_numOutputs;
	std::vector<float> m_features;
};

class LinearLayer : public Layer
{
public:
	LinearLayer(uint32_t numInputs, uint32_t numOutputs) : 
		Layer(numInputs, numOutputs)
	{
		m_weights.resize(numInputs * numOutputs);
		m_biases.resize(numOutputs);
	}
public:
	void forward(const float* inputs) override
	{
		for (uint32_t n = 0; n < m_numOutputs; ++n)
		{
			m_features[n] = m_biases[n];
			float* weights = m_weights.data() + m_numInputs * n;
			for (uint32_t i = 0; i < m_numInputs; ++i)
			{
				m_features[n] += inputs[i] * weights[i];
			}
		}
	}
private:
	std::vector<float> m_weights;
	std::vector<float> m_biases;
	//std::vector<float> m_weightDerivates;
	//std::vector<float> m_biasDerivates;
public:
};

class ActivationLayer : public Layer
{
public:
	ActivationLayer(uint32_t numInputs) :
		Layer(numInputs, numInputs)
	{
		m_derivates.resize(numInputs);
	}
protected:
	std::vector<float> m_derivates;
};

class SigmoidLayer : public ActivationLayer
{
public:
	SigmoidLayer(uint32_t numInputs) :
		ActivationLayer(numInputs)
	{}
public:
	void forward(const float* inputs) override
	{
		for (uint32_t i = 0; i < m_numOutputs; ++i)
		{
			float sigma = 1.0 / (1.0 + exp(-inputs[i]));
			m_features[i] = sigma;
			m_derivates[i] = sigma * (1.0 - sigma);
		}
	}
	void backward(const float* inputs) override
	{
	}
};

class FNN
{
public:
	void batch(float const* features, float  const* labels, uint32_t batchSize, float learningRate)
	{
		Layer* inputLayer = m_layers.front();
		uint32_t numInputs = inputLayer->numInputs();
		Layer* outputLayer = m_layers.back();
		uint32_t numOutputs = outputLayer->numOutputs();

		for (uint32_t i = 0; i < batchSize; ++i)
		{
			float const* inputs = &features[i*numInputs];
			for (Layer* layer : m_layers)
			{
				layer->forward(inputs);
				inputs = layer->outputFeatures();
			}
			m_loss->forward(inputs, &labels[numOutputs*i]);
			for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
			{
				(*it)->backward();
			}
		}
		for (Layer* layer : m_layers)
		{
			layer->update(learningRate);
		}
	}
private:

private:
	std::vector<Layer*> m_layers;
	LossFunction* m_loss;
};

void mnist2bmp()
{
}

int main()
{
	std::string path = CMAKE_SOURCE_DIR;
	mnist2bmp(path + "/data/train-images.bmp", path + "/data/train-images.idx3-ubyte");
	mnist2bmp(path + "/data/t10k-images.bmp", path + "/data/t10k-images.idx3-ubyte");
}
