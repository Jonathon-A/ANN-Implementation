package ann.implementation;

import java.util.Arrays;

public class Neuron {

    private double[] inputs;
    private final double[] weights;
    private final double[] previousWeights;
    private final double[] weightsErrorGradient;

    private double weightSum = 0;
    private double activation = 0;
    private double deltaValue = 0;

    //Initialises neuron with initial wieghts
    public Neuron(double[] weights) {
        this.weights = weights;
        this.previousWeights = Arrays.copyOf(weights, weights.length);
        this.weightsErrorGradient = new double[weights.length];
        for (int i = 0; i < weightsErrorGradient.length; i++) {
            weightsErrorGradient[i] = 0;
        }
        weightSum = 0;
    }

    //Clone neuron from exisitng neuron
    public Neuron(Neuron neuron) {
        this.weights = Arrays.copyOf(neuron.getWeights(), neuron.getWeights().length);
        this.previousWeights = Arrays.copyOf(neuron.getPreviousWeights(), neuron.getPreviousWeights().length);
        this.weightsErrorGradient = new double[weights.length];
        for (int i = 0; i < weightsErrorGradient.length; i++) {
            weightsErrorGradient[i] = 0;
        }
        weightSum = 0;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public void computeWeightSum() {
        //weightSum = ∑(weights[i] * inputs[i])
        weightSum = 0;
        for (int i = 0; i < inputs.length; i++) {
            weightSum += weights[i] * inputs[i];
        }
    }

    public double computeActivation() {
        //activation, "output" = f(weightSum)
        activation = sigmoidFunction(weightSum);
        return activation;
    }

    public void computeDeltaValue(double sampleOutput, double upsilon) {
        //For output neurons:
        // Caculate omega (for wieght decay)
        double sum = 0;
        for (double weight : weights) {
            sum += Math.pow(weight, 2);
        }
        double omega = sum / (2 * weights.length);

        //deltaValue = (sampleOutput - activation, "modelled output" + upsilon * omega) * f'(weightSum)
        deltaValue = (sampleOutput - activation + upsilon * omega) * (activation * (1 - activation));
    }

    public void computeDeltaValue(double[] nextWeights, double[] nextDeltaValues) {
        //For non-output neurons:
        //deltaValue = (sum of following weights * following delta values ) * f'(weightSum)
        double sum = 0;
        for (int i = 0; i < nextWeights.length; i++) {
            sum += (nextWeights[i] * nextDeltaValues[i]);
        }
        deltaValue = sum * (activation * (1 - activation));
    }

    public void updateWeightsAndBias(double stepSize, double momentum) {
        //Update each weight:
        //new weight = old weight + stepSize * deltaValue * respective input
        for (int i = 0; i < weights.length; i++) {
            double tempWeight = weights[i];
            weights[i] += (stepSize * deltaValue * inputs[i])
                    //Momentum
                    + (momentum * (weights[i] - previousWeights[i]));
            previousWeights[i] = tempWeight;
        }
    }

    public void sumWeightsErrorGradient() {
        //Sum error gradient for each weight
        for (int i = 0; i < weightsErrorGradient.length; i++) {
            weightsErrorGradient[i] += deltaValue * inputs[i];
        }
    }

    public void batchUpdateWeightAndBias(double stepSize, double momentum, int sampleCount) {
        //Update each weight:
        //new weight = old weight + stepSize * (error gradient / sample count)
        for (int i = 0; i < weights.length; i++) {
            double tempWeight = weights[i];
            weights[i] += (stepSize * (weightsErrorGradient[i] / sampleCount))
                    //Momentum
                    + (momentum * (weights[i] - previousWeights[i]));
            previousWeights[i] = tempWeight;
        }
        //Reset error gradient sum for each weight
        for (int i = 0; i < weightsErrorGradient.length; i++) {
            weightsErrorGradient[i] += 0;
        }
    }

    private double sigmoidFunction(double value) {
        return (1 / (1 + Math.pow(Math.E, -value)));
    }

    public double[] getWeights() {
        return weights;
    }

    public double[] getPreviousWeights() {
        return previousWeights;
    }

    public double getDeltaValue() {
        return deltaValue;
    }

    @Override
    public String toString() {
        return "Neuron{" + "inputs=" + Arrays.toString(inputs) + ", weights=" + Arrays.toString(weights) + ","
                + "\n\tweightSum=" + weightSum + ", activation=" + activation + ", deltaValue=" + deltaValue + '}';
    }
}
