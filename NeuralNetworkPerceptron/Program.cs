using System;

namespace NeuralNetworkPerceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            ErrorFunction errorFunction = new ErrorFunction((double in1, double in2) => (in1 - in2) * (in1 - in2), (double in1, double in2) => 0);
            Func<double, double> tanH = (double in1) => (Math.Exp(in1) - Math.Exp(-in1)) / (Math.Exp(in1) + Math.Exp(-in1));
            ActivationFunction activationFunction = new ActivationFunction(tanH, (double in1) => 1 - tanH(in1) * tanH(in1));
            Perceptron perceptron = new Perceptron(3, .01, errorFunction, activationFunction, new Random());

            double[][] inputs = new double[][]
            {
                new double[]{ 0, 0, 0},
                new double[]{ 0, 0, 1},
                new double[]{ 0, 1, 0},
                new double[]{ 0, 1, 1},

                //new double[]{ 1, 0, 0},
                //new double[]{ 1, 0, 1},
                //new double[]{ 1, 1, 0},
                //new double[]{ 1, 1, 1},
            };


            double[] desired = new double[]
            {
                0,
                0,
                0,
                1,

                //0,
                //1,
                //1,
                //1,
            };

            double currentError = perceptron.GetError(inputs, desired);
            while (currentError > .001)
            {
                currentError = perceptron.GetError(inputs, desired);
                Console.SetCursorPosition(0, 0);
                Console.WriteLine($"CurrentError: {currentError}");
                for(int i = 0; i < inputs.Length; i ++)
                {
                    Console.WriteLine($"Index: {i}");
                    Console.WriteLine($"\tActual: {perceptron.Train(inputs[i], desired[i])}");
                    Console.WriteLine($"\tDesired: {desired[i]}");
                }
            }
        }
    }
}
