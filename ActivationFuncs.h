#include <iostream>

class ActivationFuncs
{
	static const double alf;
public:

	static double Identity(double);
	static double IdentityDerivative(double);

	static double Sigmoid(double);
	static double SigmoidDerivative(double);

	static double ReLU(double);
	static double ReLUDerivative(double value);

	static double PReLU(double);
	static double PReLUDerivative(double value);

	static double HyperbolicTangent(double);
	static double HyperbolicTangentDerivative(double value);
};


double ActivationFuncs::Identity(double value)
{
	return value;
}
double ActivationFuncs::IdentityDerivative(double value)
{
	return 1.0;
}

double ActivationFuncs::Sigmoid(double value)
{
	return 1 / (1 + exp(-value));
}
double ActivationFuncs::SigmoidDerivative(double value)
{
	return value * (1 - value);
}

// Rectified Linear Unit
double ActivationFuncs::ReLU(double value)
{
	if (value < 0)
		return value;
	return value;
}
//Rectified Linear Unit Derivative
double ActivationFuncs::ReLUDerivative(double value)
{
	if (value < 0)
		return 0.0;
	return 1.0;
}

const double ActivationFuncs::alf = .01;
//Parametric Rectified Linear Unit 
double ActivationFuncs::PReLU(double value)
{
	if (value < 0)
		return alf * value;
	return value;
}
//Parametric Rectified Linear Unit Derivative
double ActivationFuncs::PReLUDerivative(double value)
{
	if (value < 0)
		return alf;
	return 1.0;
}


double ActivationFuncs::HyperbolicTangent(double value)
{
	return tanh(value);
}

double ActivationFuncs::HyperbolicTangentDerivative(double value)
{
	return (1 - value) * (1 + value);
}
