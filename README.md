# LittleNeuronCpp
Simple perceptron implementation.

Usage example:

```c++
#include <iostream>
#include "Neuron.h"
#include "ActivationFuncs.h"

using namespace Neurons;

int main()
{
    // neuron contstruction examples 
    //Neuron neuron(1, 0.00001, 0.000001, 0.0000001, ActivationFuncs::HyperbolicTangent, ActivationFuncs::HyperbolicTangentDerivative);
    //Neuron neuron(1, 1000, 0.00000001, 0.00000001, ActivationFuncs::PReLU, ActivationFuncs::PReLUDerivative);
    //Neuron neuron(1, 0.03, 0.00000001, 0.00000001, ActivationFuncs::Identity, ActivationFuncs::IdentityDerivative);
    //Neuron neuron(1, 0.0003, 0.00001, 0.00001, ActivationFuncs::Sigmoid, ActivationFuncs::SigmoidDerivative);

    //create a simple Neuron
    Neuron neuron(1,            // inputs count
                  0.003,        // learting rate
                  0.00000001,   // momentum
                  0.00000001,   // weights decay
                  ActivationFuncs::HyperbolicTangent, //Pointer to activation function
                  ActivationFuncs::HyperbolicTangentDerivative //Pointer to activation function derivative 
                 );

    // example set of inputs to neuron - 60 elements of 1s and 0s
    double inputs[60] = { 1,0,1,0,1,0,1,0,1,0,
                          1,0,1,0,1,0,1,0,1,0,
                          1,0,1,0,1,0,1,0,1,0,
                          1,0,1,0,1,0,1,0,1,1,
                          
                          1,0,1,0,1,0,1,0,1,0,
                          1,0,1,0,1,0,1,0,1,1 };

    // train perceptron on inputs
    for (int i = 0; i <= 8; i++) // epoch
    {
        for (int j = 0; j < 40; j++) // trainging input from 0 to 39
        {
            double arr[1] = { inputs[j] }; // prepare input array based on training data
            neuron.Feed(arr); // feed the input to neuron
            if(inputs[j] == 1) // let's say we want to get -0.9 when the input is 1
                neuron.BackPropagate(-0.9);
            else // otherwise, if input values was 0 we expect neuron to output 0.9
                neuron.BackPropagate(0.9);
            // update neuron weights based on propagated expectations
            neuron.UpdateWeights(arr);
        }
        if (i % 2 == 0) // show neuron state if it is an even epoch
            std::cout << "Current epoch is " << i << std::endl << neuron;
    }

    // print testing set of inputs
    // neuron trained on first 40 elements, so all what's left is going to be testing set
    std::cout << "[ " <<inputs[40]<<", ";
    for (int j = 41; j < 59; j++)
        std::cout << inputs[j] << ", ";
    std::cout << inputs[59] << " ]" << std::endl;

    // validate testing inputs
    int countOfRightAnswers = 0; // correct neuron's answers counter
    for (int j = 40; j < 60; j++)
    {
        double arr[1] = { inputs[j] }; // prepare input to feed to neuron
        
        neuron.Feed(arr);// feed the input

        std::cout << "Input = " << inputs[j];
        std::cout << " <--> \tOutput = " << neuron.GetActivation() << std::endl;
        
        // let's say, if we get a negative output from neuron
        // when its input = 1, then that's correct answer
        if (neuron.GetActivation() < 0 && inputs[j] == 1)
            countOfRightAnswers++;
        else if (neuron.GetActivation() > 0 && inputs[j] == 0) // if output is positive and input was 0 - correct!
            countOfRightAnswers++;
    }
    std::cout << "Number of right answers = " << countOfRightAnswers << std::endl;

}
```
