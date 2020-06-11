package code;

import java.io.*;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;

import jdk.javadoc.internal.doclets.toolkit.util.DocFinder.Output;

import java.util.Arrays;

class IrisNN {
  private static List<double[]> DataSet = new ArrayList<double[]>();
  private static List<double[]> TrainingSet = new ArrayList<double[]>();
  private static List<double[]> TestingSet = new ArrayList<double[]>();

  private static enum Set {
    TRAINING, TESTING
  }

  public static void main(String args[]) {

    // input data file
    FormatInputData("iris.data");

    // input percentage of data to use for training
    seperateDataSets(80);

    NeuralNetwork myNN = new NeuralNetwork(4, 7, 3);
    myNN.InitializeWeights();

    myNN.Train(5000, 0.05, 0.01);

    ShowVector(myNN.GetWeights(), 10);

    // System.out.println("Weights: " + Arrays.toString(myNN.GetWeights()));

    // System.out.println("Hidden Biases: " +
    // Arrays.toString(myNN.GetHiddenBiases()));
    // System.out.println("Weights: " + Arrays.toString(myNN.GetWeights()));

    double trainAcc = myNN.Accuracy(Set.TRAINING);
    System.out.println("Training Accuracy: " + trainAcc);

    double testAcc = myNN.Accuracy(Set.TESTING);
    System.out.println("Testing Accuracy: " + testAcc);

  }

  /*
   * picks random elements from the DataSet to seperate into TrainingSet and
   * TestingSet. Amount of data to be used for TrainingSet is passed in as
   * parameter
   */
  private static void seperateDataSets(double percentageTraining) {
    Random rand = new Random(); // instance of random class

    List<double[]> TempDataSet = new ArrayList<double[]>(DataSet);

    for (int i = 0; i < DataSet.size() * (percentageTraining / 100); i++) {
      int randInt = rand.nextInt(TempDataSet.size());
      TrainingSet.add(TempDataSet.get(randInt));
      TempDataSet.remove(randInt);
    }
    // what remains is testing data
    TestingSet = new ArrayList<double[]>(TempDataSet);
  }

  static void ShowVector(double[] vector, int valsPerRow) {
    for (int i = 0; i < vector.length; ++i) {
      if (i % valsPerRow == 0)
        System.out.println("");
      System.out.print(String.format("%.2f", vector[i]));
    }
    System.out.println("");
  }

  /*
   * convert 5.1, 3.5, 1.4, 0.2, Iris setosa into 5.1, 3.5, 1.4, 0.2, 0, 0, 1 Iris
   * setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0
   */
  private static void FormatInputData(String filePath) {

    try {
      File myObj = new File(filePath);
      Scanner myReader = new Scanner(myObj);
      while (myReader.hasNextLine()) {
        String data = myReader.nextLine();
        String dataValues[] = data.split(",");
        double[] singleDataPoint = new double[7];
        for (int i = 0; i < dataValues.length; i++) {
          if (dataValues[i].isEmpty())
            continue;
          if (i < 4) {
            // measured values
            singleDataPoint[i] = Double.parseDouble(dataValues[i]);
          } else {
            // category
            if (dataValues[i].equals("Iris-setosa")) {
              singleDataPoint[4] = 0;
              singleDataPoint[5] = 0;
              singleDataPoint[6] = 1;
            } else if (dataValues[i].equals("Iris-versicolor")) {
              singleDataPoint[4] = 0;
              singleDataPoint[5] = 1;
              singleDataPoint[6] = 0;
            } else if (dataValues[i].equals("Iris-virginica")) {
              singleDataPoint[4] = 1;
              singleDataPoint[5] = 0;
              singleDataPoint[6] = 0;
            }
          }
        }
        if (!(singleDataPoint[4] == 0 && singleDataPoint[5] == 0 && singleDataPoint[6] == 0))
          DataSet.add(singleDataPoint);
      }
      myReader.close();
    } catch (FileNotFoundException e) {
      System.out.println("An error occurred.");
      e.printStackTrace();
    }
  }

  private static class NeuralNetwork {
    private Random random;
    private int InputCount;
    private int HiddenCount;
    private int OutputCount;

    private double[] Inputs;

    private double[][] inputHiddenWeights;
    private double[] HiddenBiases;
    private double[] HiddenOutputs;

    private double[][] outputHiddenWeights;
    private double[] OutputBiases;

    private double[] Outputs;

    private double[] OutputGradients;
    private double[] HiddenGradients;

    // MOVE TO TRAINING METHOD
    private double[][] inputHiddenPreviousWeightsChange; // ihPrevWeightsDelta;
    private double[] hiddenPreviousBiasesChange; // hPrevBiasesDelta;
    private double[][] hiddenOutputPreviousWeightChange; // hoPrevWeightsDelta;
    private double[] outputPreviousBiasesChange; // oPrevBiasesDelta;

    // constructor
    public NeuralNetwork(int InputCount_, int HiddenCount_, int OutputCount_) {
      random = new Random(0); // to initialize weights and shuffle

      // initialize object
      InputCount = InputCount_;
      HiddenCount = HiddenCount_;
      OutputCount = OutputCount_;

      Inputs = new double[InputCount];

      inputHiddenWeights = TWODArray(InputCount, HiddenCount);
      HiddenBiases = new double[HiddenCount];
      HiddenOutputs = new double[HiddenCount];

      outputHiddenWeights = TWODArray(HiddenCount, OutputCount_);
      OutputBiases = new double[OutputCount];

      Outputs = new double[OutputCount];

      HiddenGradients = new double[HiddenCount];
      OutputGradients = new double[OutputCount];

      inputHiddenPreviousWeightsChange = TWODArray(InputCount, HiddenCount);
      hiddenPreviousBiasesChange = new double[HiddenCount];
      hiddenOutputPreviousWeightChange = TWODArray(HiddenCount, OutputCount);
      outputPreviousBiasesChange = new double[OutputCount];
    } // constructor

    // create and fill a TWODArray of Double
    private double[][] TWODArray(int rows, int cols) // helper for ctor
    {
      double[][] output = new double[rows][];
      for (int i = 0; i < output.length; i++)
        output[i] = new double[cols];
      return output;
    }

    public void SetWeights(double[] weights) {
      int WeightsCount = (InputCount * HiddenCount) + (HiddenCount * OutputCount) + HiddenCount + OutputCount;

      int weightsCounter = 0;

      for (int i = 0; i < InputCount; i++)
        for (int j = 0; j < HiddenCount; j++)
          inputHiddenWeights[i][j] = weights[weightsCounter++];
      for (int i = 0; i < HiddenCount; i++)
        HiddenBiases[i] = weights[weightsCounter++];
      for (int i = 0; i < HiddenCount; i++)
        for (int j = 0; j < OutputCount; j++)
          outputHiddenWeights[i][j] = weights[weightsCounter++];
      for (int i = 0; i < OutputCount; i++)
        OutputBiases[i] = weights[weightsCounter++];
    }

    // small random values
    public void InitializeWeights() {
      int WeightsCount = (InputCount * HiddenCount) + (HiddenCount * OutputCount) + HiddenCount + OutputCount;
      double[] initialWeights = new double[WeightsCount];
      double lowerBound = -0.01;
      double upperBound = 0.01;
      for (int i = 0; i < initialWeights.length; i++)
        initialWeights[i] = (upperBound - lowerBound) * random.nextDouble() + lowerBound;
      SetWeights(initialWeights);
    }

    public double[] GetHiddenBiases() {
      return HiddenBiases;
    }

    public double[] GetOutputBiases() {
      return OutputBiases;
    }

    public double[] GetWeights() {
      int WeightsCount = (InputCount * HiddenCount) + (HiddenCount * OutputCount) + HiddenCount + OutputCount;
      double[] output = new double[WeightsCount];
      int weightsCounter = 0;
      for (int i = 0; i < inputHiddenWeights.length; i++)
        for (int j = 0; j < inputHiddenWeights[0].length; j++)
          output[weightsCounter++] = inputHiddenWeights[i][j];
      for (int i = 0; i < HiddenBiases.length; i++)
        output[weightsCounter++] = HiddenBiases[i];
      for (int i = 0; i < outputHiddenWeights.length; i++)
        for (int j = 0; j < outputHiddenWeights[0].length; j++)
          output[weightsCounter++] = outputHiddenWeights[i][j];
      for (int i = 0; i < OutputBiases.length; i++)
        output[weightsCounter++] = OutputBiases[i];
      return output;
    }

    private double[] ComputeOutputs(double[] xValues) {
      double[] HiddenTotal = new double[HiddenCount];
      double[] OutputTotal = new double[OutputCount];

      for (int i = 0; i < xValues.length; i++)
        Inputs[i] = xValues[i];

      for (int j = 0; j < HiddenCount; j++)
        for (int i = 0; i < InputCount; i++)
          HiddenTotal[j] += Inputs[i] * inputHiddenWeights[i][j];

      for (int i = 0; i < HiddenCount; i++)
        HiddenTotal[i] += HiddenBiases[i];

      for (int i = 0; i < HiddenCount; i++)
        HiddenOutputs[i] = HyperTanFunction(HiddenTotal[i]);

      for (int j = 0; j < OutputCount; j++)
        for (int i = 0; i < HiddenCount; i++)
          OutputTotal[j] += HiddenOutputs[i] * outputHiddenWeights[i][j];

      for (int i = 0; i < OutputCount; i++)
        OutputTotal[i] += OutputBiases[i];

      double[] OutputSMax = SMax(OutputTotal);

      System.arraycopy(OutputSMax, 0, Outputs, 0, OutputSMax.length);

      double[] output = new double[OutputCount];

      System.arraycopy(Outputs, 0, output, 0, Outputs.length);

      return output;
    }

    private double HyperTanFunction(double x) {
      if (x < -20.0)
        return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0)
        return 1.0;
      else
        return Math.tanh(x);
    }

    private double[] SMax(double[] OutputTotal) {
      double max = OutputTotal[0];
      for (int i = 0; i < OutputTotal.length; i++)
        if (OutputTotal[i] > max)
          max = OutputTotal[i];

      double scale = 0.0;
      for (int i = 0; i < OutputTotal.length; i++)
        scale += Math.exp(OutputTotal[i] - max);

      double[] output = new double[OutputTotal.length];
      for (int i = 0; i < OutputTotal.length; i++)
        output[i] = Math.exp(OutputTotal[i] - max) / scale;

      return output;
    }

    private void UpdateWeights(double[] tValues, double learnRate, double alpha) {
      for (int i = 0; i < OutputGradients.length; i++) {
        double derivative = (1 - Outputs[i]) * Outputs[i];
        OutputGradients[i] = derivative * (tValues[i] - Outputs[i]);
      }

      for (int i = 0; i < HiddenGradients.length; i++) {
        double derivative = (1 - HiddenOutputs[i]) * (1 + HiddenOutputs[i]);
        double sum = 0.0;
        for (int j = 0; j < OutputCount; j++) {
          double x = OutputGradients[j] * outputHiddenWeights[i][j];
          sum += x;
        }
        HiddenGradients[i] = derivative * sum;
      }

      for (int i = 0; i < inputHiddenWeights.length; i++) {
        for (int j = 0; j < inputHiddenWeights[0].length; j++) {
          double delta = learnRate * HiddenGradients[j] * Inputs[i];
          inputHiddenWeights[i][j] += delta;
          inputHiddenWeights[i][j] += alpha * inputHiddenPreviousWeightsChange[i][j];
          inputHiddenPreviousWeightsChange[i][j] = delta;
        }
      }

      for (int i = 0; i < HiddenBiases.length; i++) {
        double delta = learnRate * HiddenGradients[i] * 1.0;
        HiddenBiases[i] += delta;
        HiddenBiases[i] += alpha * hiddenPreviousBiasesChange[i];
        hiddenPreviousBiasesChange[i] = delta;
      }

      for (int i = 0; i < outputHiddenWeights.length; i++) {
        for (int j = 0; j < outputHiddenWeights[0].length; j++) {
          double delta = learnRate * OutputGradients[j] * HiddenOutputs[i];
          outputHiddenWeights[i][j] += delta;
          outputHiddenWeights[i][j] += alpha * hiddenOutputPreviousWeightChange[i][j];
          hiddenOutputPreviousWeightChange[i][j] = delta;
        }
      }

      for (int i = 0; i < OutputBiases.length; i++) {
        double delta = learnRate * OutputGradients[i] * 1.0;
        OutputBiases[i] += delta;
        OutputBiases[i] += alpha * outputPreviousBiasesChange[i];
        outputPreviousBiasesChange[i] = delta;
      }
    }

    public void Train(int maxEpoch, double learnRate, double alpha) {
      int epoch = 0;
      double[] xValues = new double[InputCount];
      double[] tValues = new double[OutputCount];

      int[] sequence = new int[TrainingSet.size()];
      for (int i = 0; i < sequence.length; i++)
        sequence[i] = i;

      while (epoch < maxEpoch) {
        double mse = MeanSquaredError();
        if (mse < 0.020)
          break;

        Shuffle(sequence);
        for (int i = 0; i < TrainingSet.size(); i++) {
          int idx = sequence[i];
          System.arraycopy(TrainingSet.get(idx), 0, xValues, 0, InputCount);
          System.arraycopy(TrainingSet.get(idx), InputCount, tValues, 0, OutputCount);
          ComputeOutputs(xValues);
          UpdateWeights(tValues, learnRate, alpha);
        }
        epoch++;
      }
    }

    private void Shuffle(int[] sequence) {
      for (int i = 0; i < sequence.length; i++) {

        int r = (int) ((Math.random() * (sequence.length - i)) + i);

        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    private double MeanSquaredError() {
      double sumSquaredError = 0.0;
      double[] xValues = new double[InputCount];
      double[] tValues = new double[OutputCount];

      for (int i = 0; i < TrainingSet.size(); i++) {
        System.arraycopy(TrainingSet.get(i), 0, xValues, 0, InputCount);
        System.arraycopy(TrainingSet.get(i), InputCount, tValues, 0, OutputCount);

        double[] yValues = ComputeOutputs(xValues);
        for (int j = 0; j < OutputCount; ++j) {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }

      return sumSquaredError / TrainingSet.size();
    }

    public double Accuracy(Set set) {
      List<double[]> mySet = new ArrayList<double[]>();
      if (set == Set.TESTING)
        mySet = TestingSet;
      else if (set == Set.TRAINING)
        mySet = TrainingSet;

      int CorrectCount = 0;
      int WrongCount = 0;
      double[] xValues = new double[InputCount];
      double[] tValues = new double[OutputCount];
      double[] yValues;

      for (int i = 0; i < mySet.size(); i++) {

        System.arraycopy(mySet.get(i), 0, xValues, 0, InputCount);
        System.arraycopy(mySet.get(i), InputCount, tValues, 0, OutputCount);

        yValues = ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues);

        if (tValues[maxIndex] == 1.0)
          CorrectCount++;
        else
          WrongCount++;
      }
      return (CorrectCount * 1.0) / (CorrectCount + WrongCount);
    }

    private int MaxIndex(double[] vector) {
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.length; ++i) {
        if (vector[i] > biggestVal) {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }

  }

}