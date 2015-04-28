#include <iostream>
#include <QtGui/QSurfaceFormat>
#include <ui/texturewidget.h>
#include <QtWidgets/QFileDialog>

//static const texture map declaration
GLfloat const TextureWidget::texcoord[8] = {
    0.0f,0.0f,
    0.0f,1.0f,
    1.0f,1.0f,
    1.0f,0.0f,
  };

//static index data declaration
GLubyte const TextureWidget::indexData[6] = {0,1,2,0,2,3};


TextureWidget::TextureWidget(QWidget *par) :
                        QOpenGLWidget(par),
                        m_glsl_program(nullptr),
                        m_texture(nullptr),
                        m_texture_buffer(nullptr),
                        m_offset(vec2(0.0,0.0)),
                        m_scale(1),
                        last_x(0),
                        last_y(0)

{
    //setting wanted opengl version
    QSurfaceFormat qformat;
    qformat.setDepthBufferSize(24);
    qformat.setVersion(4, 3);
    qformat.setSwapInterval(0);

    setFormat(qformat);

    //setting up the frame count
    m_frameCount = 0;
    m_currentTime = QTime::currentTime();
    m_pastTime = m_currentTime;

    //setting up shortcuts
    reset_short= new QShortcut(QKeySequence("Ctrl+r"), this);
    QObject::connect(reset_short, SIGNAL(activated()), this, SLOT(reset_user_transform()));
    fit_short= new QShortcut(QKeySequence("Ctrl+f"), this);
    QObject::connect(fit_short, SIGNAL(activated()), this, SLOT(fit_to_screen()));




}


////////////////// FILE HANDLING /////////////////
void TextureWidget::open_bmp_from_disk(const char * path)
{
    this->show();
    //first make sure we cleanup memory before re-allocating
    clean_up_texture_data();

    //creating and loading the texture to display
    m_texture = new Bitmap;
    m_texture->open(path);

    //creating the texture buffer
    m_texture_buffer = new QOpenGLTexture(QOpenGLTexture::Target2D);
    m_texture_buffer->create();
    m_texture_buffer->bind();

    //manually setting up the texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //querying texture size
    float w,h;
    w = (float)m_texture->get_width();
    h = (float)m_texture->get_height();

    //manually copying the texure data on the gpu
    glTexImage2D(GL_TEXTURE_2D,
                 0, GL_RGB,
                 w, h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 m_texture->getRawData());
    //generating mip maps
    m_texture_buffer->generateMipMaps();

    //creating the correct positional buffer data
    create_vertex_data((int)w,(int)h);
    //fitting the image on screen
    fit_to_screen();
}

void TextureWidget::upload_cpu_buffer(uint8_t * buffer)
{
    
    float w = (float)m_texture->get_width();
    float h = (float)m_texture->get_height();
    //manually setting up the texture parameters
    m_texture_buffer->bind();
   
    glTexImage2D(GL_TEXTURE_2D,
                 0, GL_RGB,
                 w, h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 buffer);
    m_texture_buffer->generateMipMaps();
    update();
}


void TextureWidget::open()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "/home/", tr("Image Files (*.bmp)"));
    if (fileName.isNull())
    {
        return;
    }
    //TODO CHECK FOR EXCEPTION THROW
    open_bmp_from_disk( fileName.toLocal8Bit().constData());



}

Bitmap * TextureWidget::get_image_data() const
{
    return m_texture;
}

////////////////// DATA MANAGEMENT ////////////////
void TextureWidget::clean_up_texture_data()
{
    makeCurrent();
    if (m_texture)
    {
        delete m_texture;
    }

    if (m_texture_buffer)
    {
        m_texture_buffer->destroy();
        delete m_texture_buffer;
    }
    m_texture = nullptr;
    m_texture_buffer = nullptr;
}

void TextureWidget::create_vertex_data(const int &width,
                                       const int &height)
{
    makeCurrent();
    m_vao1->bind();
    //creating and setupping the vertex buffer
    float positionData[12] = {0.0f, 0.0f, 0.0f,
                               0.0f, (float)height, 0.0f,
                               (float)width, (float)height, 0.0f,
                            (float)width, 0, 0.0f};

    m_positionBuffer= (QOpenGLBuffer(QOpenGLBuffer::VertexBuffer));
    m_positionBuffer.create();
    m_positionBuffer.setUsagePattern( QOpenGLBuffer::StaticDraw );
    m_positionBuffer.bind();
    m_positionBuffer.allocate(positionData, 12 * (int)sizeof(float));
    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL );

}

////////////////////// UTILITIES /////////////////

void TextureWidget::reset_user_transform()
{
    m_scale =1;
    m_offset= vec2(0,0);
    update();
}

void TextureWidget::fit_to_screen()
{
    if (!m_texture)
    {
        return;
    }
    m_offset= vec2(0,0);
    //set it to the ration
    int winW;
    winW = this->width();
    float ratio = float(winW)/float(m_texture->get_width());

    m_scale = ratio;

    update();
}

void TextureWidget::computeFps()
{
    //  Increase frame count
    m_frameCount++;

    //  Get the number of milliseconds since glutInit called
    //  (or first call to glutGet(GLUT ELAPSED TIME)).
    m_currentTime = QTime::currentTime();

    //  Calculate time passed
    m_timeInterval = m_pastTime.msecsTo(m_currentTime);

    if(m_timeInterval > 1000)
    {
        //  calculate the number of frames per second
        m_fps = m_frameCount / (int)((float)m_timeInterval / 1000.0f);
        //  Set time
        m_pastTime = m_currentTime;

        //  Reset frame count
        m_frameCount = 0;
        std::cout<<m_fps<<" FPS"<<std::endl;
    }
}

///////////////////////// OPENGL ///////////////////

void TextureWidget::initializeGL()
{
    //initializing all the opengl function and context
    initializeOpenGLFunctions();

    //creatting a new GLSL_program
    m_glsl_program = new GLSL_program;

    //loading and compiling vertex shaders
    m_glsl_program->load_vertex_shader("../textureViewer/texture.vert");
    m_glsl_program->load_fragment_shader("../textureViewer/texture.frag");
    //setting the program as active
    m_glsl_program->use();

    //crating vertex array object
    m_vao1 = new QOpenGLVertexArrayObject( this );
    m_vao1->create();
    m_vao1->bind();

    //crating and allocating the index buffer
    m_indexBuffer= (QOpenGLBuffer(QOpenGLBuffer::IndexBuffer));
    m_indexBuffer.create();
    m_indexBuffer.setUsagePattern( QOpenGLBuffer::StaticDraw );
    m_indexBuffer.bind();
    m_indexBuffer.allocate(indexData, 6 * (int)sizeof(GLubyte));

    //crating and allocating the text coordinates
    m_texture_coord= (QOpenGLBuffer(QOpenGLBuffer::VertexBuffer));
    m_texture_coord.create();
    m_texture_coord.setUsagePattern( QOpenGLBuffer::StaticDraw );
    m_texture_coord.bind();
    m_texture_coord.allocate(texcoord, 8 * (int)sizeof(GLfloat));
    glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL );

}

void TextureWidget::resizeGL(int w, int h)
{
    //setting new size for the viewport and
    //re-calculating the ortho matrix for proj
    glViewport(0,0,GLsizei(w),GLsizei(h));
    m_ortho = glm::ortho<GLfloat>( 0.0f, (GLfloat)w, (GLfloat)h, 0.0f, 1.0f, -1.0f );
}

void TextureWidget::paintGL()
{
    if (!m_texture_buffer || !m_texture)
    {
        return;
    }
    // computeFps();

    glClear(GL_COLOR_BUFFER_BIT);
    //binding the needed buffers
    m_vao1->bind();
    m_texture_buffer->bind();

    //computing final matrix as a result of
    //the user transformation and projection
    //matrix
    m_world = glm::mat4(1.0f);
    m_world = glm::scale(m_world,glm::vec3(m_scale,m_scale,1));
    //we are panning around and we are compensating for the scale factor
    m_world = glm::translate(m_world,glm::vec3(m_offset.x/m_scale,
                                               m_offset.y/m_scale,
                                               0.0));
    m_final = m_ortho * m_world;

    //passing the matrix to the shaders
    m_glsl_program->set_shader_attribute("ortho",m_final);

    //enabling needed attributes and paint
    glEnableVertexAttribArray(0);  // Vertex position
    glEnableVertexAttribArray(1);  // Text maps
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (GLvoid*)0);


    float w,h;
    w = (float)m_texture->get_width();
    h = (float)m_texture->get_height();
    create_vertex_data((int)w,(int)h);

}

void TextureWidget::opengl_version_log()
{
    const GLubyte *renderer = glGetString( GL_RENDERER );
    const GLubyte *vendor = glGetString( GL_VENDOR );
    const GLubyte *version = glGetString( GL_VERSION );
    const GLubyte *glslVersion = glGetString( GL_SHADING_LANGUAGE_VERSION );

    GLint major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    //log
    printf("GL Vendor            : %s\n", vendor);
    printf("GL Renderer          : %s\n", renderer);
    printf("GL Version (string)  : %s\n", version);
    printf("GL Version (integer) : %d.%d\n", major, minor);
    printf("GLSL Version         : %s\n", glslVersion);

}


////////////////////// MOUSE EVENTS SETUP ////////////
void TextureWidget::mouseMoveEvent(QMouseEvent  *e)
{

    //extracting the mouse positon
    int posX, posY;
    posX = e->pos().x();
    posY = e->pos().y();

    //computing the delta vector respect
    //previous position
    int deltaX = posX- last_x;
    int deltaY = posY- last_y;
    if(e->buttons() == Qt::MidButton)
    {
        //setting the new pan value
        m_offset+=  vec2(deltaX,deltaY);
        //forcing paint
        update();

    }
    else if (e->buttons() == Qt::RightButton)
    {
        //delcaring delta vec and computing its length
        vec2 delta(deltaX,deltaY);
        float len =glm::length(delta);

        //we perform a dot product to find if we scale
        //up or down
        if (glm::dot(vec2(1.0f,0),delta)>0)
        {
            m_scale+=(len/1000.0f);
            // m_offset+=  vec2(-m_scale * m_texture->get_width()
            //                 ,m_scale * m_texture->get_height());             
        }else
        {
            m_scale-=(len/1000.0f);
            // m_offset-=  vec2(m_scale * m_texture->get_width()
            //                 ,m_scale * m_texture->get_height());            
        }

        //forcing paint
        update();
    }
    //updating previous position
    last_x =posX;
    last_y =posY;
}
void TextureWidget::mousePressEvent(QMouseEvent  *e)
{
    //storing the previos pose in case
    //the user starts to drag
    last_x = e->pos().x();
    last_y = e->pos().y();
}

/////////////// DESTRUCTOR ////////////////
TextureWidget::~TextureWidget()
{
    //making sure the context is current
    makeCurrent();

    //deleting the allocated program
    if (m_glsl_program)
    {
    delete m_glsl_program;
    }


    if (m_vao1)
    {
        delete m_vao1;
    }

    clean_up_texture_data();
}



