--primeros joins de aumento de datos, aumentado con tabla usuarios

select a.*,b.edad,substr(b.fecha_alta,1,4) as anio_alta,b.tipo
from ratings_train a,usuarios b
where a.id_usuario=b.id_usuario;

--creacion de tabla de cantidad de seguidores por usuario

select id_usuario,count(id_usuario_seguido) as cant_seguidores
into usuarios_seguidores
from siguiendo
group by id_usuario;