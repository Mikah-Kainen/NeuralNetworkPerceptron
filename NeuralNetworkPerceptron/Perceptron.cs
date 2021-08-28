using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkPerceptron
{

    public class Perceptron
    {
        public double LearningRate { get; set; }
        public double bias;
        public double[] weights;
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
            double returnVal = bias;
            for(int i = 0; i < values.Length; i ++)
            {
                returnVal += values[i] * weights[i];
            }   
            returnVal = activationFunction.Function(returnVal);
            return returnVal;
        }

        private double ComputeNoA(double[] values)
        {
            double returnVal = bias;
            for (int i = 0; i < values.Length; i++)
            {
                returnVal += values[i] * weights[i];
            }
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

        private double[] GetChangeValues(double[] values, double desired)
        {
            double[] changeValues = new double[weights.Length + 1];
            double nonactivatedOut = ComputeNoA(values);
            double activatedOut = activationFunction.Function(nonactivatedOut);

            double currentError = errorFunction.Derivitive(activatedOut, desired);
            double currentDerivitive = activationFunction.Derivitive(nonactivatedOut);
            double partialDerivitive;
            for (int i = 0; i < values.Length; i++)
            {

                partialDerivitive = currentError * currentDerivitive * values[i];
                changeValues[i] = -1 * LearningRate * partialDerivitive;
            }
            partialDerivitive = currentError * currentDerivitive;
            changeValues[^1] = -1 * LearningRate * partialDerivitive;

            return changeValues;
        }

        public double Train(double[] values, double desired)
        {
            double[] changeValues = GetChangeValues(values, desired);

            for (int i = 0; i < weights.Length; i ++)
            {
                weights[i] += changeValues[i];
            }
            bias += changeValues[^1];
            return errorFunction.Function(Compute(values), desired);
        }

        public double Train(double[][] values, double[] desired)
        {
            double[][] changeValuesArray = new double[desired.Length][];
            
            for(int i = 0; i < desired.Length; i ++)
            {
                changeValuesArray[i] = GetChangeValues(values[i], desired[i]);
            }

            for(int i = 0; i < changeValuesArray.Length; i ++)
            {
                for(int x = 0; x < changeValuesArray[i].Length-1; x ++)
                {
                    weights[x] += changeValuesArray[i][x];
                }
                bias += changeValuesArray[i][^1];
            }
            return GetError(values, desired);
        }

    }
}
