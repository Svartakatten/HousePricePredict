using Microsoft.ML.Data;

namespace PricePrediction.Models
{
    public class HousingData
    {
        [LoadColumn(0)]
        public float Size { get; set; }

        [LoadColumn(1)]
        public float Bedrooms { get; set; }

        [LoadColumn(2)]
        public float Price { get; set; }
    }

    public class HousingPricePrediction
    {
        public float Score { get; set; }
    }
}