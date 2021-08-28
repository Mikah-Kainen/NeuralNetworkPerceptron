using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace NeuralNetworkPerceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            ErrorFunction errorFunction = new ErrorFunction((double output, double desired) => Math.Pow(desired - output, 2), (double output, double desired) => -2 * (desired - output));
            Func<double, double> tanH = (double in1) => (Math.Exp(in1) - Math.Exp(-in1)) / (Math.Exp(in1) + Math.Exp(-in1));
            ActivationFunction activationFunction = new ActivationFunction(tanH, (double in1) => 1 - tanH(in1) * tanH(in1));
            ActivationFunction binaryStep = new ActivationFunction((double in1) => in1 < 0 ? 0 : 1, (double in1) => 1);
            ActivationFunction test = new ActivationFunction((double x) => x, (double x) => 1);
            
            Perceptron perceptron = new Perceptron(2, .0005, errorFunction, binaryStep, new Random());

            double[][] inputs = new double[][]
            {
                //new double[]{ 0, 0, 0},
                //new double[]{ 0, 1, 0},
                //new double[]{ 0, 0, 1},
                //new double[]{ 0, 1, 1},

                //new double[]{ 1, 0, 0},
                //new double[]{ 1, 0, 1},
                //new double[]{ 1, 1, 0},
                //new double[]{ 1, 1, 1},

                new double[]{0, 0},
                new double[]{0, 1},
                new double[]{1, 0},
                new double[]{1, 1},
            };


            double[] desired = new double[]
            {
                //0,
                //0,
                //0,
                //1,

                //0,
                //1,
                //1,
                //1,

                0,
                1,
                1,
                0,
            };

            Stack<double[]> debugStack = new Stack<double[]>();
            Stack<double> biasStack = new Stack<double>();

            double currentError = perceptron.GetError(inputs, desired);
            while (currentError > .001)
            {
                currentError = perceptron.GetError(inputs, desired);
                Console.SetCursorPosition(0, 0);
                Console.WriteLine($"CurrentError: {currentError}");
                perceptron.Train(inputs, desired);
                for(int i = 0; i < inputs.Length; i ++)
                {
                    Console.WriteLine($"Index: {i}");
                    Console.WriteLine($"\tActual: {perceptron.Compute(inputs[i])}");
                    Console.WriteLine($"\tDesired: {desired[i]}");
                }
                Console.WriteLine($"Bias: {perceptron.bias}");
                debugStack.Push(perceptron.weights.ToArray());
                biasStack.Push(perceptron.bias);
                //Stopwatch stopwatch = new Stopwatch();
                //stopwatch.Restart();
                //while (stopwatch.ElapsedMilliseconds < 10) ;
            }

            List<double[]> debugList = new List<double[]>();
            List<double> biasList = new List<double>();
            for(int i = 0; i < debugStack.Count; i ++)
            {
                debugList.Add(debugStack.Pop());
            }
            for (int i = 0; i < 50; i ++)
            {
                biasList.Add(biasStack.Pop());
            }

            int count = 0;
            foreach(double[] values in debugList)
            {
                count++;
                foreach (double value in values)
                {
                    if(value.Equals(double.NaN))
                    {

                    }
                }
            }

        }
    }
}
