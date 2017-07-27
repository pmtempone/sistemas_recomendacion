--primeros joins de aumento de datos, aumentado con tabla usuarios

select a.*,b.edad,substr(b.fecha_alta,1,4) as anio_alta,b.tipo
from ratings_train a,usuarios b
where a.id_usuario=b.id_usuario;

--