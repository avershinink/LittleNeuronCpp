#include "Neuron.h"
#include <iostream>

Neuron::Neuron(int inputsCount):
	learningRate_(0.0050),
	momentum_    (0.0002),
	decay_       (0.0001)
{
	inputsCount_ = inputsCount;
	Init();
}

Neuron::Neuron(int inputsCount, double learningRate, double momentum, double decay) :
	inputsCount_(inputsCount),
	learningRate_(learningRate),
	momentum_(momentum),
	decay_(decay)
{
	Init();
}

Neuron::Neuron(int inputsCount, double learningRate, double momentum, double decay, NeuronFunc ActivationFunc, NeuronFunc ActivationDerivativeFunc):
	inputsCount_(inputsCount),
	learningRate_(learningRate),
	momentum_(momentum),
	decay_(decay)
{
	Init();
	this->ActivationFunc = ActivationFunc;
	this->ActivationDerivativeFunc = ActivationDerivativeFunc;
}

Neuron::~Neuron()
{
	//std::cout << "Destructing Neuron" << std::endl;
	delete[] weights_;
}

void Neuron::Init()
{
	weights_ = new double[inputsCount_];

	InitWeights();

	bias_ = 1;
	biasDelta_ = 0;
	biasPrevDelta_ = 0;

	delta_ = 0;
	prevDelta_ = 0;
	net_sum_ = 0;
	activation_ = 0;
}

void Neuron::Feed(double* inputs)
{
	net_sum_ = 0;
	for (int i = 0; i < inputsCount_; i++)
		net_sum_ += inputs[i] * weights_[i];

	net_sum_ += bias_ * biasWeight_;

	activation_ = ActivationFunc(net_sum_);
}

void Neuron::Telemetry(int accuracy) const
{
	std::cout << std::endl;
	std::cout << "========================================" << std::endl;
	std::printf("\tBias = %.9f", bias_);
	std::cout << std::endl;
	std::printf("\t\tbiasDelta_ = %.9f", biasDelta_);
	std::cout << std::endl;
	std::printf("\t\tbiasPrevDelta_ = %.9f", biasPrevDelta_);
	std::cout << std::endl;
	
	std::cout << "\tWeights: " << std::endl;
	std::cout << "\t\t";
	PrintWeights(std::cout);

	std::printf("\tDelta = %.9f", delta_);
	std::cout << std::endl;
	std::printf("\tPrev Delta = %.9f", prevDelta_);
	std::cout << std::endl;
	std::printf("\tACTIVATION = %.9f", activation_);
	std::cout << std::endl;
	std::cout << "========================================" << std::endl;
}

void Neuron::Telemetry(void) const
{
	Telemetry(16);
}

void Neuron::PrintWeights(std::ostream &DstStream) const
{
	DstStream << "[ " << weights_[0];
	for (int i = 1; i < inputsCount_; i++)
		DstStream << ", " << weights_[i];
	DstStream << "]" << std::endl;
}

void Neuron::InitWeights(void)
{
	for (int i = 0; i < inputsCount_; i++)
		weights_[i] = (rand() % 1000) / 1000.0;
}


void Neuron::BackPropagate(double targetOutput)
{
	double deviation = ActivationDerivativeFunc(activation_);
	delta_ = deviation * (targetOutput - activation_);
	biasDelta_ = deviation * 1;
}

void Neuron::UpdateWeights(double* inputs)
{
	double learningDelta = 0.0;
	double biasLearningDelta = 0.0;
	for (int i = 0; i < inputsCount_; i++)
	{
		 learningDelta = learningRate_ * delta_ * inputs[i];
		 weights_[i] += learningDelta;
		 weights_[i] += momentum_ * prevDelta_;
		 weights_[i] -= decay_ * weights_[i];
		 prevDelta_ = learningDelta;

		 biasLearningDelta = learningRate_ * delta_ * 1;
		 bias_ += biasLearningDelta;
		 bias_ += momentum_ * biasPrevDelta_;
		 bias_ -= decay_ * bias_;
		 biasPrevDelta_ = learningDelta;
	}
}

double Neuron::GetActivation(void) const
{
	return activation_;
}


std::ostream& operator<<(std::ostream &DstStream, Neuron &PrjNeuron)
{
	DstStream << std::endl;
	DstStream << "========================================" << std::endl;
	DstStream << "\tBias = " << PrjNeuron.bias_ << std::endl;
	DstStream << "\t\tbiasDelta_ = " << PrjNeuron.biasDelta_ << std::endl;
	DstStream << "\t\tbiasPrevDelta_ = " << PrjNeuron.biasPrevDelta_ << std::endl;

	DstStream << "\tWeights: " << std::endl;
	DstStream << "\t\t";
	PrjNeuron.PrintWeights(DstStream);

	DstStream << "\tDelta = " << PrjNeuron.delta_ << std::endl;
	DstStream << "\tPrev Delta = " << PrjNeuron.prevDelta_ << std::endl;
	DstStream << "\tACTIVATION = " << PrjNeuron.activation_ << std::endl;
	DstStream << "========================================" << std::endl;
	return DstStream;
}