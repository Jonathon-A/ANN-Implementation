package ann.implementation;

public final class Sample {

    private final double[] data;

    public Sample(double[] data) {
        this.data = data;
    }

    public double[] getData() {
        return data;
    }

    @Override
    public String toString() {
        String str = "Sample:\t ";
        for (int i = 0; i < data.length - 1; i++) {
            str += String.format("\tpredictor %d: %.4f", i, data[i]);
        }
        str += String.format("\tpridictand: %.4f", data[data.length - 1]);
        return str;
    }

}
