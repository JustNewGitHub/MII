using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

//====================================================
// Описание работы классов и методов исходника на:
// https://www.interestprograms.ru
// Исходные коды программ и игр
//====================================================

namespace WpfMoveWindow
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
            _offsetPoint = new Point(posCursor.X - ffElement.Margin.Left, posCursor.Y - ffElement.Margin.Top);

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

                    Thickness margin = new(p.X - _offsetPoint.X, p.Y - _offsetPoint.Y, 0, 0);
                    ffElement.Margin = margin;
                }
            }
        }

        private void FF_MouseUp(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            _canMove = false;
           e.MouseDevice.Capture(null);
        }

        #endregion

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Click!");
        }
    }
}
