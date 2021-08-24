using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkPerceptron
{

    public class Perceptron
    {
        public double LearningRate { get; set; }
        double bias;
        double[] weights;
        ErrorFunction errorFunction;
        ActivationFunction activationFunction;

        public Perceptron()
        {

        }

    }
}
