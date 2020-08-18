#include <iostream>

class ActivationFuncs
{
	static const double alf;
public:
	static double PReLU(double);
	static double PReLUDerivative(double value);

	static double HyperbolicTangent(double);
	static double HyperbolicTangentDerivative(double value);
};

const double ActivationFuncs::alf = .01;

double ActivationFuncs::PReLU(double value)
{
	if (value < 0)
		return alf * value;
	return value;
}
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
