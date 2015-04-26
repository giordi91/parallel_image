#ifndef TEXTUREWIDGET_H
#define TEXTUREWIDGET_H

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <QtWidgets/QOpenGLWidget>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLTexture>
#include <QtGui/QCloseEvent>
#include <QtCore/QTime>
#include <QtCore/QTimer>
#include <QtWidgets/QShortcut>

#include <ui/glsl_program.h>
#include <core/bitmap.h>

using glm::vec2;
using glm::mat4x4;
class TextureWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
Q_OBJECT


public:
    /**
     * @brief TextureWidget, used to display texture with opengl
     * @param parent the pointer at parent widget
     */
    explicit TextureWidget(QWidget *parent = 0);
    /**
     * @brief The destructor
     */
    virtual ~TextureWidget();
    /**
     * @brief Overritten method for initializing opengl context
     */
    void initializeGL();
    /**
     * @brief Method automatically called for resize of opengl
     * @param w: the new widget width
     * @param h: the new widget height
     */
    void resizeGL(int w, int h);
    /**
     * @brief Opengl paint function
     */
    void paintGL();
    /**
     * @brief Logging out specs of GPU and opengl used
     */
    void opengl_version_log();
    /**
     * @brief load an image from file
     * @param path: the path to the file
     * @throw invalid_argument: if path is invalid
     */
    void open_bmp_from_disk(const char * path);

    /**
     * @brief create the vertex data based on the image size
     * @param width: the width of the image
     * @param height: the height of the image
     */
    void create_vertex_data(const int &width,
                            const int &height);
    /**
     * This function returns a pointer to the image class
     */
    Bitmap * get_image_data() const; 


protected:
    /**
     * @brief Mouse event for manipulating the texture,
     * @description: the widgets uses two mouse buttons to work
     * the position and size of the texture, the right mouse button
     * allows to scale the texture, left -> scale up, right -> scale down
     * the middle mouse is used for panning around
     * @param event: argument passed by qt
     */
    void mouseMoveEvent(QMouseEvent  *event);
    /**
     * @brief event used to track the start of drag
     * @param event: argument passed by qt
     */
    void mousePressEvent(QMouseEvent  *event);

    /**
     * @brief Function used internally to compute fps
     */
    void computeFps();
    /**
     * @brief Frees all the memory allocated for the texture
     */
    void clean_up_texture_data();




private:
    //the glsl_program instance used to load/compile/link
    //the shaders
    GLSL_program * m_glsl_program;
    //class for loading bitmaps
    Bitmap * m_texture;
    //the position buffer of our picture
    // float  positionData[9];
    //static index buffer used for the image
    static const GLubyte   indexData[6];
    //static texture coordinate for the image
    static  const GLfloat texcoord[8];

    //internal buffers
    QOpenGLBuffer m_positionBuffer;
    QOpenGLBuffer m_indexBuffer;
    QOpenGLBuffer m_texture_coord;
    QOpenGLVertexArrayObject * m_vao1;
    QOpenGLTexture   * m_texture_buffer;

    //internal values
    //the offset applied to the image
    vec2 m_offset;
    //the scale applied to the image
    float m_scale;
    //the projection matrix
    mat4x4 m_ortho;
    //the world matrix
    mat4x4 m_world;
    //the resulting matrix
    mat4x4 m_final;

    //recordingo of the previous mouse position
    int last_x;
    int last_y;

    //fps counter vars
    int m_fps;
    QTime m_currentTime;
    QTime m_pastTime;
    int m_frameCount;
    int m_timeInterval;

    //shortcuts pointers
    QShortcut *reset_short;
    QShortcut *fit_short;


signals:

public slots:
    /**
     * @brief SLOT that ask for path and open image
     */
    void open();
    /**
     * @brief SLOT resets the user manipulation
     */
    void reset_user_transform();
    /**
     * @brief SLOT fits the current image to the screen
     */
    void fit_to_screen();

};

#endif // TEXTUREWIDGET_H
