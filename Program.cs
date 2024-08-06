using Microsoft.ML;
using PricePrediction.Models;
using System;
using System.IO;

namespace PricePrediction
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Uppdatera sökvägen till datasetet för felsökning
            var dataPath = Path.Combine(Environment.CurrentDirectory, "housing_data.csv");
            Console.WriteLine($"Looking for file at: {dataPath}");

            // Kontrollera om filen finns
            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"File not found: {dataPath}");
            }
            else
            {
                Console.WriteLine($"File found: {dataPath}");
            }

            // Ladda data
            Console.WriteLine("Laddar data");
            var data = context.Data.LoadFromTextFile<HousingData>(dataPath, separatorChar: ',', hasHeader: true);

            // Definiera dataprocessen
            Console.WriteLine("Definerar processen");
            var pipeline = context.Transforms.Concatenate("Features", nameof(HousingData.Size), nameof(HousingData.Bedrooms))
                .Append(context.Transforms.CopyColumns("Label", nameof(HousingData.Price)))
                .Append(context.Regression.Trainers.Sdca());

            // Träna modellen
            Console.WriteLine("Tränar modellen");
            var model = pipeline.Fit(data);

            // Använd modellen för att göra en förutsägelse
            var predictionEngine = context.Model.CreatePredictionEngine<HousingData, HousingPricePrediction>(model);

            var input = new HousingData { Size = 850, Bedrooms = 2 };
            var prediction = predictionEngine.Predict(input);

            Console.WriteLine($"Predicted Price: {prediction.Score}");
        }
    }
}