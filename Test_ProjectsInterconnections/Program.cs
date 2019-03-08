using IronPython.Hosting;
using Microsoft.Scripting.Hosting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test_ProjectsInterconnections
{
    class Program
    {
        static void Main(string[] args)
        {
            int yNumber = 10;
            ScriptEngine engine = Python.CreateEngine();
            ScriptScope scope = engine.CreateScope();
            scope.SetVariable("y", yNumber);
            engine.ExecuteFile(@"C:\Users\swite\YandexDisk\diplom\ResearchPaperAnalysis\Test_PythonApplication\Test_PythonApplication.py",
                scope);
            dynamic xNumber = scope.GetVariable("x");
            dynamic zNumber = scope.GetVariable("z");
            Console.WriteLine("Сумма {0} и {1} равна: {2}", xNumber, yNumber, zNumber);
        }

    }
}

