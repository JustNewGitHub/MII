using System.Windows;
using System.Windows.Controls;

//====================================================
// Описание работы классов и методов исходника на:
// https://www.interestprograms.ru
// Исходные коды программ и игр
//====================================================

namespace WpfMoveElementsCanvas
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }


        #region Функция перемещения элементов

        int countZ = 0;
        bool _canMove = false;
        Point _offsetPoint = new(0, 0);
        private void FF_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            _canMove = true;
            countZ++;

            FrameworkElement ffElement = (FrameworkElement)sender;
         
            Grid.SetZIndex(ffElement, countZ);


            Point posCursor = e.MouseDevice.GetPosition(this);
            //_offsetPoint = new Point(posCursor.X - ffElement.Margin.Left, posCursor.Y - ffElement.Margin.Top);
            _offsetPoint = new Point(posCursor.X - Canvas.GetLeft(ffElement), posCursor.Y - Canvas.GetTop(ffElement));

            // Чтобы курсор не оторвался от фигуры
            e.MouseDevice.Capture(ffElement);
        }

        private void FF_MouseMove(object sender, System.Windows.Input.MouseEventArgs e)
        {
            if (_canMove == true)
            {
                FrameworkElement ffElement = (FrameworkElement)sender;

                if (e.MouseDevice.Captured == ffElement)
                {
                    Point p = e.MouseDevice.GetPosition(this);

                    //Thickness margin = new(p.X - _offsetPoint.X, p.Y - _offsetPoint.Y, 0, 0);
                    //ffElement.Margin = margin;
                    Canvas.SetLeft(ffElement, p.X - _offsetPoint.X);
                    Canvas.SetTop(ffElement, p.Y - _offsetPoint.Y);
                }
            }
        }

        private void FF_MouseUp(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            _canMove = false;
            e.MouseDevice.Capture(null);
        }

        #endregion
    }
}
