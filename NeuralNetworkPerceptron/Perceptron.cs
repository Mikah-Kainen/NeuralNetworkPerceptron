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
        Random random;

        public Perceptron(int amountOfInputs, double learningRate, ErrorFunction errorFunction, ActivationFunction activationFunction, Random random)
        {
            weights = new double[amountOfInputs];
            LearningRate = learningRate;
            this.errorFunction = errorFunction;
            this.activationFunction = activationFunction;
            this.random = random;
            Randomize(0, 1);
        }

        public void Randomize(double min, double max)
        {
            for(int i = 0; i < weights.Length; i ++)
            {
                weights[i] = random.NextDouble(min, max);
            }
            bias = random.NextDouble(min, max);
        }

        public double Compute(double[] values)
        {
            double returnVal = 0;
            for(int i = 0; i < values.Length; i ++)
            {
                returnVal += values[i] * weights[i];
            }
            returnVal += bias;
            returnVal = activationFunction.Function(returnVal);
            return returnVal;
        }

        public double[] Compute(double[][] values)
        {
            double[] returnArray = new double[values.Length];
            for (int i = 0; i < returnArray.Length; i++)
            {
                returnArray[i] = Compute(values[i]);
            }
            return returnArray;
        }

        public double GetError(double[][] values, double[] desired)
        {
            double returnValue = 0;
            for (int i = 0; i < values.Length; i++)
            {
                returnValue += errorFunction.Function(Compute(values[i]), desired[i]);
            }
            returnValue /= values.Length;
            return returnValue;
        }

        public double Train(double[] values, double desired)
        {
            double[] changeValues = new double[weights.Length];
            for(int i = 0; i < changeValues.Length; i ++)
            {
                double weightsPlusBias = bias;
                foreach(double weight in weights)
                {
                    weightsPlusBias += weight;
                }
                double partialDerivitive = errorFunction.Function(Compute(values), desired) * activationFunction.Derivitive(weightsPlusBias) * values[i];
                changeValues[i] = LearningRate * -partialDerivitive;
            }

            for(int i = 0; i < changeValues.Length; i ++)
            {
                weights[i] += changeValues[i];
            }
            return errorFunction.Function(Compute(values), desired);
        }

        public double[] Train(double[][] values, double[] desired)
        {
            double[] returnArray = new double[values.Length];
            for(int i = 0; i < returnArray.Length; i ++)
            {
                returnArray[i] = Train(values[i], desired[i]);
            }
            return returnArray;
        }

    }
}
