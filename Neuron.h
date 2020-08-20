#include <iostream>

namespace Neurons
{
	class Neuron
	{
		typedef double(*NeuronFunc) (double);

		friend std::ostream& operator<<(std::ostream&, Neuron&);

	public:
		Neuron(void);
		Neuron(int);
		//@parem1<int>    -- number of inputs
		//@parem2<double> -- learningRate
		//@parem3<double> -- momentum
		//@parem4<double> -- decay
		Neuron(int, double, double, double);
		//@parem1<int>    -- number of inputs
		//@parem2<double> -- learningRate
		//@parem3<double> -- momentum
		//@parem4<double> -- decay
		//@parem5<Func> -- Activation function
		//@parem6<Func> -- Derivative of the activation function 
		Neuron(int, double, double, double, NeuronFunc, NeuronFunc);

		Neuron(const Neuron &);

		Neuron(const Neuron * const);

		~Neuron();

		void Feed(double*);
		//@param targetOutput -- neuron expected aim output
		void BackPropagate(double);
		//@parem inputs -- neuron entries 
		void UpdateWeights(double*);
		//@param<int> -- output value accurancy
		void InitWeights(void);
		double GetActivation(void) const;

		Neuron & operator=(const Neuron & rhs);

	private:
		int inputsCount_;
		double learningRate_;
		double momentum_;
		double decay_;

		double net_sum_;
		double* weights_;
		double activation_;

		double bias_;
		double biasWeight_;
		double biasDelta_;
		double biasPrevDelta_;

		double delta_;
		double prevDelta_;

		NeuronFunc ActivationFunc;
		NeuronFunc ActivationDerivativeFunc;
		//double ActivationFunc(double);
		//double ActivationDerivativeFunc(double);

		void Init();
		void PrintWeights(std::ostream &) const;
		void copy(const Neuron &);
	};
}