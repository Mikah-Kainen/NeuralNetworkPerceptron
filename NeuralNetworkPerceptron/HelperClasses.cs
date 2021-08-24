using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkPerceptron
{
    public class ActivationFunction
    {
        Func<double, double> function;
        Func<double, double> derivitive;

        public ActivationFunction(Func<double, double> function, Func<double, double> derivitive)
        {
            this.function = function;
            this.derivitive = derivitive;
        }

        public double Function(double input)
        {
            return function(input);
        }

        public double Derivitive(double input)
        {
            return derivitive(input);
        }
    }

    public class ErrorFunction
    {
        Func<double, double, double> function;
        Func<double, double, double> derivitive;

        public ErrorFunction(Func<double, double, double> function, Func<double, double, double> derivitive)
        {
            this.function = function;
            this.derivitive = derivitive;
        }

        public double Function(double input, double desired)
        {
            return function(input, desired);
        }

        public double Derivitive(double input, double desired)
        {
            return derivitive(input, desired);
        }
    }
}
